"""
Ethnic Language Corpus Loader
==============================
Loads parallel text corpora for Ede and Bahnar languages,
building sentence-aligned pairs for Vietnamese↔Ethnic MT.

Sources supported:
  1. IJERT Ede-Vietnamese bilingual corpus (from PDF reference)
  2. Bahnar language resources
  3. Custom CSV/TSV bilingual files
  4. Synthetic backtranslation pairs
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


@dataclass
class BilingualPair:
    """A single sentence-aligned bilingual pair."""
    source_lang: str        # e.g. "vi", "ede", "bah"
    target_lang: str
    source_text: str
    target_text: str
    domain: str = "general"
    source_file: str = ""
    confidence: float = 1.0


# ── Built-in seed vocabulary (from PDF references) ────────────────────────────

# Ede-Vietnamese seed vocabulary extracted from IJERT paper
EDE_VI_SEED: list[tuple[str, str]] = [
    # (Ede, Vietnamese)
    ("ih", "tôi"),
    ("ih", "mình"),
    ("drei", "ba"),
    ("buôn", "buôn làng"),
    ("êga", "anh ấy"),
    ("năm", "năm"),
    ("mniê", "phụ nữ"),
    ("êkei", "đàn ông"),
    ("dlan", "đường"),
    ("hma", "rẫy"),
    ("lăn", "đất"),
    ("êa", "nước"),
    ("boh", "quả"),
    ("ana", "con"),
    ("ama", "cha"),
    ("ami", "mẹ"),
    ("adei", "em"),
    ("abô", "ông"),
    ("aduôn", "bà"),
    ("sang", "nhà"),
    ("ƀuôn", "làng"),
    ("klei", "lời"),
    ("mtao", "vua"),
    ("yang", "thần"),
    ("hdruôm", "sách"),
    ("čiăng", "muốn"),
    ("hriăm", "học"),
    ("blŭ", "nói"),
    ("čhưng", "nghe"),
    ("mưng", "từ"),
    ("kơ", "cho"),
    ("leh", "đã"),
    ("amăng", "trong"),
    ("gah", "phía"),
    ("anăp", "trước"),
    ("tlâo", "ba"),
    ("pă", "bốn"),
    ("êma", "năm"),
    ("năm pă", "chín"),
    ("pluh", "mười"),
]

# Bahnar-Vietnamese seed vocabulary from bahna.pdf reference
BAHNAR_VI_SEED: list[tuple[str, str]] = [
    # (Bahnar, Vietnamese)
    ("mâh", "một"),
    ("bar", "hai"),
    ("pơ", "ba"),
    ("puăt", "bốn"),
    ("rơ̆m", "năm"),
    ("năm", "sáu"),
    ("pơjuh", "bảy"),
    ("sa pă", "tám"),
    ("chín", "chín"),
    ("pluh", "mười"),
    ("tơ̆", "tôi"),
    ("mơ̆ng", "từ"),
    ("amăng", "trong"),
    ("tơpa", "trời"),
    ("ia", "nước"),
    ("blang", "đồng ruộng"),
    ("kơ̆nă", "con"),
    ("mah", "vàng"),
    ("anăm", "ăn"),
    ("mơ-ang", "uống"),
    ("pơtŭ", "ngủ"),
    ("tơ̆m", "cây"),
    ("bơnga", "hoa"),
    ("hơmŭ", "lúa"),
    ("yang", "thần"),
    ("rơnơh", "làng"),
    ("sang", "nhà"),
    ("hơdơ̆ng", "đường"),
    ("ama", "cha"),
    ("mă", "mẹ"),
    ("kơ̆čơl", "con trai"),
    ("kơ̆nŭ", "con gái"),
    ("pô", "người"),
    ("hơdip", "sống"),
    ("hơmat", "chết"),
    ("klah", "đẹp"),
    ("rơgơ̆m", "xấu"),
    ("kơ-iăng", "nói"),
    ("tơ̆ng", "nghe"),
    ("glăh", "thấy"),
]


class EthnicCorpusLoader:
    """
    Loads and processes ethnic language corpora (Ede, Bahnar).

    Supported formats:
    - CSV/TSV with columns: source_lang, target_lang, source, target
    - JSON lines: {"vi": "...", "ede": "..."}
    - Plain text pairs (alternating lines)
    - Built-in seed vocabulary
    """

    SUPPORTED_LANGS = {"vi", "ede", "bah"}

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        include_seed: bool = True,
        max_pairs: Optional[int] = None,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.include_seed = include_seed
        self.max_pairs = max_pairs
        self._pairs: list[BilingualPair] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> DatasetDict:
        """Load all corpora and return a DatasetDict split by language pair."""
        self._pairs = []
        if self.include_seed:
            self._load_seed_vocabulary()
        if self.corpus_dir and self.corpus_dir.exists():
            self._load_from_directory()
        logger.info(f"Loaded {len(self._pairs):,} bilingual pairs total")
        return self._to_dataset_dict()

    def load_ede_vi(self) -> Dataset:
        """Return Ede↔Vietnamese pairs as HuggingFace Dataset."""
        self.load_all()
        pairs = [p for p in self._pairs if {p.source_lang, p.target_lang} == {"vi", "ede"}]
        return self._pairs_to_dataset(pairs)

    def load_bahnar_vi(self) -> Dataset:
        """Return Bahnar↔Vietnamese pairs as HuggingFace Dataset."""
        self.load_all()
        pairs = [p for p in self._pairs if {p.source_lang, p.target_lang} == {"vi", "bah"}]
        return self._pairs_to_dataset(pairs)

    def generate_backtranslation_pairs(
        self,
        translator_fn,
        source_texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[BilingualPair]:
        """
        Generate synthetic pairs via backtranslation.
        translator_fn: callable(text, src, tgt) -> str
        """
        pairs = []
        for text in source_texts:
            try:
                translated = translator_fn(text, source_lang, target_lang)
                back = translator_fn(translated, target_lang, source_lang)
                pairs.append(BilingualPair(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_text=text,
                    target_text=translated,
                    domain="synthetic_backtranslation",
                    confidence=0.7,
                ))
            except Exception as e:
                logger.warning(f"Backtranslation failed for '{text[:30]}...': {e}")
        logger.info(f"Generated {len(pairs)} backtranslation pairs ({source_lang}→{target_lang})")
        return pairs

    # ── Private: Load ─────────────────────────────────────────────────────────

    def _load_seed_vocabulary(self) -> None:
        """Load built-in seed vocabulary as sentence pairs."""
        for ede_word, vi_word in EDE_VI_SEED:
            self._pairs.append(BilingualPair(
                source_lang="ede", target_lang="vi",
                source_text=ede_word, target_text=vi_word,
                domain="vocabulary", source_file="seed",
            ))
            # Add reverse
            self._pairs.append(BilingualPair(
                source_lang="vi", target_lang="ede",
                source_text=vi_word, target_text=ede_word,
                domain="vocabulary", source_file="seed",
            ))
        for bah_word, vi_word in BAHNAR_VI_SEED:
            self._pairs.append(BilingualPair(
                source_lang="bah", target_lang="vi",
                source_text=bah_word, target_text=vi_word,
                domain="vocabulary", source_file="seed",
            ))
            self._pairs.append(BilingualPair(
                source_lang="vi", target_lang="bah",
                source_text=vi_word, target_text=bah_word,
                domain="vocabulary", source_file="seed",
            ))
        logger.info(
            f"Loaded {len(EDE_VI_SEED)*2} Ede-Vi + {len(BAHNAR_VI_SEED)*2} Bahnar-Vi seed pairs"
        )

    def _load_from_directory(self) -> None:
        """Load all supported corpus files from corpus_dir."""
        for path in sorted(self.corpus_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix in (".csv", ".tsv"):
                    self._load_csv(path)
                elif suffix == ".jsonl":
                    self._load_jsonl(path)
                elif suffix == ".json":
                    self._load_json(path)
                elif suffix == ".txt":
                    self._load_txt_pairs(path)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    def _load_csv(self, path: Path) -> None:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
        required = {"source_lang", "target_lang", "source", "target"}
        if not required.issubset(df.columns):
            logger.warning(f"Skipping {path.name}: missing columns {required - set(df.columns)}")
            return
        for _, row in df.iterrows():
            self._pairs.append(BilingualPair(
                source_lang=str(row["source_lang"]),
                target_lang=str(row["target_lang"]),
                source_text=str(row["source"]),
                target_text=str(row["target"]),
                domain=str(row.get("domain", "general")),
                source_file=str(path),
            ))
        logger.info(f"Loaded {len(df)} pairs from {path.name}")

    def _load_jsonl(self, path: Path) -> None:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Support {"vi": "...", "ede": "..."} or {"vi": "...", "bah": "..."}
                for src, tgt in [("vi", "ede"), ("vi", "bah"), ("ede", "vi"), ("bah", "vi")]:
                    if src in obj and tgt in obj:
                        self._pairs.append(BilingualPair(
                            source_lang=src, target_lang=tgt,
                            source_text=str(obj[src]),
                            target_text=str(obj[tgt]),
                            source_file=str(path),
                        ))
                        count += 1
                        break
        logger.info(f"Loaded {count} pairs from {path.name}")

    def _load_json(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                for src, tgt in [("vi", "ede"), ("vi", "bah"), ("ede", "vi"), ("bah", "vi")]:
                    if src in obj and tgt in obj:
                        self._pairs.append(BilingualPair(
                            source_lang=src, target_lang=tgt,
                            source_text=str(obj[src]),
                            target_text=str(obj[tgt]),
                            source_file=str(path),
                        ))
                        break

    def _load_txt_pairs(self, path: Path) -> None:
        """Load alternating line pairs: even=source, odd=target."""
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        # Detect language from filename
        name = path.stem.lower()
        if "ede" in name:
            src_lang, tgt_lang = "ede", "vi"
        elif "bah" in name or "bahnar" in name:
            src_lang, tgt_lang = "bah", "vi"
        else:
            return
        for i in range(0, len(lines) - 1, 2):
            self._pairs.append(BilingualPair(
                source_lang=src_lang, target_lang=tgt_lang,
                source_text=lines[i].strip(),
                target_text=lines[i + 1].strip(),
                source_file=str(path),
            ))

    # ── Private: Convert ──────────────────────────────────────────────────────

    def _pairs_to_dataset(self, pairs: list[BilingualPair]) -> Dataset:
        if self.max_pairs:
            pairs = pairs[: self.max_pairs]
        records = [
            {
                "source_lang": p.source_lang,
                "target_lang": p.target_lang,
                "source_text": p.source_text,
                "target_text": p.target_text,
                "domain": p.domain,
                "confidence": p.confidence,
            }
            for p in pairs
        ]
        return Dataset.from_list(records)

    def _to_dataset_dict(self) -> DatasetDict:
        ede_vi = [p for p in self._pairs if {p.source_lang, p.target_lang} == {"vi", "ede"}]
        bah_vi = [p for p in self._pairs if {p.source_lang, p.target_lang} == {"vi", "bah"}]
        return DatasetDict({
            "ede_vi": self._pairs_to_dataset(ede_vi),
            "bah_vi": self._pairs_to_dataset(bah_vi),
        })

    def print_statistics(self) -> None:
        from collections import Counter
        pairs_by_lang = Counter(f"{p.source_lang}→{p.target_lang}" for p in self._pairs)
        domains = Counter(p.domain for p in self._pairs)
        print("\n=== Ethnic Corpus Statistics ===")
        print(f"Total pairs: {len(self._pairs):,}")
        print("\nPairs by language:")
        for lang_pair, cnt in pairs_by_lang.most_common():
            print(f"  {lang_pair}: {cnt:,}")
        print("\nPairs by domain:")
        for domain, cnt in domains.most_common():
            print(f"  {domain}: {cnt:,}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ethnic Corpus Loader CLI")
    parser.add_argument("--corpus-dir", default=None, help="Path to corpus directory")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--no-seed", action="store_true", help="Skip built-in seed vocab")
    args = parser.parse_args()

    loader = EthnicCorpusLoader(
        corpus_dir=args.corpus_dir,
        include_seed=not args.no_seed,
    )
    dd = loader.load_all()
    if args.stats:
        loader.print_statistics()
    for name, ds in dd.items():
        print(f"DatasetDict['{name}']: {len(ds):,} pairs")


if __name__ == "__main__":
    main()
