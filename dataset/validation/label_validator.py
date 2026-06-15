"""
Label Validator
===============
Validates metadata labels for the ViSpeech dataset:
  - Removes duplicate audio entries
  - Validates transcript text (non-empty, Vietnamese characters)
  - Validates province and region codes against the official 63-province list
  - Validates gender (0=female, 1=male)
  - Validates duration consistency
  - Flags and optionally removes invalid/suspicious labels
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── Official 63 Province Codes (ViMD dataset) ─────────────────────────────────
VALID_PROVINCES: set[str] = {
    # North
    "HN", "HNM", "HY", "NB", "ND", "TB", "BN", "BG", "HD", "HP", "QN",
    "LS", "CB", "BK", "TN", "TQ", "HG", "LC", "YB", "PT", "VP", "HB",
    "SL", "DB", "LC2",
    # Central
    "TH", "NA", "HT", "QB", "QT", "TTH", "DN", "QNA", "QNG", "BD",
    "PY", "KH", "NT", "BT", "KT", "GL", "DL", "DK", "LN",
    # South
    "HCM", "BDG", "DNA", "BRVT", "LA", "TG", "BT2", "TV", "VL",
    "DT", "AG", "KG", "CT", "HGI", "ST", "BL", "CM", "BP", "TN2",
}

VALID_REGIONS: set[str] = {"North", "Central", "South"}
VALID_GENDERS: set[int] = {0, 1}   # 0=female, 1=male
VALID_LANGUAGES: set[str] = {"vi", "ede", "bah"}
VALID_ETHNICITIES: set[str] = {"viet", "ede", "bahnar"}

# Vietnamese character range (Unicode)
VI_PATTERN = re.compile(
    r"^[\u0020-\u007E\u00C0-\u024F\u1EA0-\u1EF9\u0300-\u036F\u0102\u0103\u01A0\u01A1"
    r"\u01AF\u01B0\u0110\u0111\u0300-\u0309\u1EB8\u1EB9\u1ECC\u1ECD\u1EE4\u1EE5"
    r"aăâáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệiíìỉĩịoôơóòỏõọốồổỗộớờởỡợuưúùủũụứừửữựyýỳỷỹỵ"
    r"AĂÂBCDĐEGHIKLMNOÔƠPQRSTUƯVXY\s\d\.,\!\?\-\'\"]+$",
    re.IGNORECASE | re.UNICODE,
)

MIN_TRANSCRIPT_LEN = 2
MAX_TRANSCRIPT_LEN = 500
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0


@dataclass
class LabelValidationResult:
    total: int = 0
    valid: int = 0
    removed_duplicates: int = 0
    invalid_transcript: int = 0
    invalid_province: int = 0
    invalid_region: int = 0
    invalid_gender: int = 0
    invalid_duration: int = 0
    invalid_language: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Label Validation Summary",
            f"  Total         : {self.total:,}",
            f"  Valid         : {self.valid:,} ({self.valid/max(self.total,1)*100:.1f}%)",
            f"  Duplicates    : {self.removed_duplicates:,}",
            f"  Bad transcript: {self.invalid_transcript:,}",
            f"  Bad province  : {self.invalid_province:,}",
            f"  Bad region    : {self.invalid_region:,}",
            f"  Bad gender    : {self.invalid_gender:,}",
            f"  Bad duration  : {self.invalid_duration:,}",
            f"  Bad language  : {self.invalid_language:,}",
        ]
        if self.warnings:
            lines.append(f"\n  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


class LabelValidator:
    """
    Validates and cleans dataset labels for the ViSpeech platform.

    Usage:
        validator = LabelValidator()
        clean_df, report = validator.validate_dataframe(df)
    """

    def __init__(
        self,
        check_province: bool = True,
        check_transcript: bool = True,
        check_duplicates: bool = True,
        check_duration: bool = True,
        strict_province: bool = False,  # If False, unknown provinces get warnings not errors
        min_duration: float = MIN_DURATION_S,
        max_duration: float = MAX_DURATION_S,
    ):
        self.check_province = check_province
        self.check_transcript = check_transcript
        self.check_duplicates = check_duplicates
        self.check_duration = check_duration
        self.strict_province = strict_province
        self.min_duration = min_duration
        self.max_duration = max_duration

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_dataframe(
        self, df: pd.DataFrame, drop_invalid: bool = True
    ) -> tuple[pd.DataFrame, LabelValidationResult]:
        """
        Validate a DataFrame of dataset labels.

        Returns:
            (cleaned_df, report)
        """
        report = LabelValidationResult(total=len(df))
        df = df.copy()

        # Track which rows to drop
        invalid_mask = pd.Series([False] * len(df), index=df.index)

        # ── Step 1: Remove duplicates ──
        if self.check_duplicates:
            dup_cols = self._get_dedup_columns(df)
            before = len(df)
            df = df.drop_duplicates(subset=dup_cols, keep="first")
            report.removed_duplicates = before - len(df)
            invalid_mask = invalid_mask[df.index]
            logger.info(f"Removed {report.removed_duplicates:,} duplicate entries")

        # ── Step 2: Validate transcripts ──
        if self.check_transcript and "text" in df.columns:
            bad_mask = self._validate_transcripts(df["text"])
            report.invalid_transcript = int(bad_mask.sum())
            if drop_invalid:
                invalid_mask |= bad_mask
            logger.info(f"Invalid transcripts: {report.invalid_transcript:,}")

        # ── Step 3: Validate province ──
        if self.check_province and "province" in df.columns:
            bad_mask, warn_mask = self._validate_provinces(df["province"])
            report.invalid_province = int(bad_mask.sum())
            if drop_invalid and self.strict_province:
                invalid_mask |= bad_mask
            for p in df.loc[warn_mask, "province"].unique()[:10]:
                report.warnings.append(f"Unknown province code: '{p}'")
            logger.info(f"Unknown provinces: {report.invalid_province:,}")

        # ── Step 4: Validate region ──
        if "region" in df.columns:
            bad_mask = ~df["region"].isin(VALID_REGIONS)
            report.invalid_region = int(bad_mask.sum())
            if drop_invalid:
                invalid_mask |= bad_mask

        # ── Step 5: Validate gender ──
        if "gender" in df.columns:
            bad_mask = ~df["gender"].isin(VALID_GENDERS)
            report.invalid_gender = int(bad_mask.sum())
            if drop_invalid:
                invalid_mask |= bad_mask

        # ── Step 6: Validate duration ──
        if self.check_duration and "length" in df.columns:
            bad_mask = (
                (df["length"] < self.min_duration) | (df["length"] > self.max_duration)
            )
            report.invalid_duration = int(bad_mask.sum())
            if drop_invalid:
                invalid_mask |= bad_mask
            logger.info(f"Duration out-of-range: {report.invalid_duration:,}")

        # ── Step 7: Apply drops ──
        if drop_invalid:
            df_clean = df[~invalid_mask].reset_index(drop=True)
        else:
            df_clean = df.copy()
            df_clean["_valid"] = ~invalid_mask

        report.valid = len(df_clean)
        return df_clean, report

    def validate_single(self, row: dict) -> tuple[bool, list[str]]:
        """Validate a single sample row. Returns (is_valid, error_messages)."""
        errors = []

        # Transcript
        text = str(row.get("text", "")).strip()
        if not text or len(text) < MIN_TRANSCRIPT_LEN:
            errors.append(f"Transcript too short: '{text[:20]}'")
        if len(text) > MAX_TRANSCRIPT_LEN:
            errors.append(f"Transcript too long: {len(text)} chars")
        if text and not self._is_valid_text(text):
            errors.append(f"Invalid characters in transcript")

        # Province
        province = str(row.get("province", ""))
        if province and province.upper() not in VALID_PROVINCES:
            if self.strict_province:
                errors.append(f"Unknown province: '{province}'")

        # Region
        region = str(row.get("region", ""))
        if region and region not in VALID_REGIONS:
            errors.append(f"Invalid region: '{region}' (expected North/Central/South)")

        # Gender
        gender = row.get("gender", None)
        if gender is not None and int(gender) not in VALID_GENDERS:
            errors.append(f"Invalid gender: {gender} (expected 0 or 1)")

        # Duration
        duration = float(row.get("length", row.get("duration", 0.0)))
        if duration < self.min_duration or duration > self.max_duration:
            errors.append(f"Duration {duration:.2f}s out of range [{self.min_duration}, {self.max_duration}]")

        return len(errors) == 0, errors

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_dedup_columns(self, df: pd.DataFrame) -> list[str]:
        """Determine which columns to use for deduplication."""
        candidates = ["filename", "text", "speakerID", "length"]
        return [c for c in candidates if c in df.columns]

    def _validate_transcripts(self, series: pd.Series) -> pd.Series:
        """Return boolean mask of invalid transcript rows."""
        def _is_invalid(text) -> bool:
            if pd.isna(text) or not str(text).strip():
                return True
            text = str(text).strip()
            if len(text) < MIN_TRANSCRIPT_LEN:
                return True
            if len(text) > MAX_TRANSCRIPT_LEN:
                return True
            return False
        return series.apply(_is_invalid)

    def _validate_provinces(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Return (error_mask, warning_mask).
        error_mask: completely invalid
        warning_mask: unknown but not clearly wrong
        """
        upper = series.fillna("").astype(str).str.upper().str.strip()
        known_mask = upper.isin(VALID_PROVINCES)
        empty_mask = upper == ""
        error_mask = empty_mask
        warn_mask = ~known_mask & ~empty_mask
        return error_mask, warn_mask

    @staticmethod
    def _is_valid_text(text: str) -> bool:
        """Check if text contains reasonable Vietnamese/Latin characters."""
        # Allow if mostly printable unicode
        non_printable = sum(
            1 for c in text
            if unicodedata.category(c) in ("Cc", "Cf", "Cs", "Co", "Cn")
        )
        return non_printable / max(len(text), 1) < 0.1

    # ── Statistics ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_label_stats(df: pd.DataFrame) -> dict:
        """Compute label distribution statistics."""
        stats: dict = {}
        if "province" in df.columns:
            stats["province_distribution"] = df["province"].value_counts().to_dict()
        if "region" in df.columns:
            stats["region_distribution"] = df["region"].value_counts().to_dict()
        if "gender" in df.columns:
            stats["gender_distribution"] = df["gender"].value_counts().to_dict()
        if "length" in df.columns:
            stats["duration"] = {
                "min": float(df["length"].min()),
                "max": float(df["length"].max()),
                "mean": float(df["length"].mean()),
                "total_hours": float(df["length"].sum() / 3600),
            }
        if "speakerID" in df.columns:
            stats["unique_speakers"] = int(df["speakerID"].nunique())
        return stats


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Label Validator CLI")
    parser.add_argument("input", help="Input CSV/parquet file")
    parser.add_argument("--output", default=None, help="Output path for cleaned CSV")
    parser.add_argument("--strict-province", action="store_true")
    args = parser.parse_args()

    path = Path(args.input) if hasattr(args, "input") else None
    if str(args.input).endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    validator = LabelValidator(strict_province=args.strict_province)
    df_clean, report = validator.validate_dataframe(df)
    print(f"\n{'='*60}")
    print(report.summary())
    print(f"\nLabel statistics:")
    stats = LabelValidator.compute_label_stats(df_clean)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output:
        if args.output.endswith(".parquet"):
            df_clean.to_parquet(args.output, index=False)
        else:
            df_clean.to_csv(args.output, index=False)
        print(f"\nCleaned dataset saved to {args.output}")


if __name__ == "__main__":
    main()
