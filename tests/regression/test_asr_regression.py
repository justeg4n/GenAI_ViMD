"""
Regression Tests — ASR/Dialect Benchmark Suite
================================================
Runs WER/CER/accuracy benchmarks on the test split.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

TEST_AUDIO_DIR = Path(os.getenv("TEST_AUDIO_DIR", "tests/fixtures/audio"))
TEST_MANIFEST_PATH = Path(os.getenv("TEST_MANIFEST", "tests/fixtures/test_manifest.json"))


def load_test_manifest() -> list[dict]:
    """Load the test manifest (list of {audio_path, transcript, province, region})."""
    if TEST_MANIFEST_PATH.exists():
        with open(TEST_MANIFEST_PATH) as f:
            return json.load(f)
    # Return stub data if no manifest
    return [
        {
            "audio_path": None,
            "transcript": "Xin chào thế giới",
            "province": "HAN",
            "region": "North",
            "ethnicity": "viet",
        }
    ]


# ── WER Utility ───────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) using edit distance."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])

    return d[len(ref_words), len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER)."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])

    return d[len(ref_chars), len(hyp_chars)] / len(ref_chars)


# ── WER Tests ─────────────────────────────────────────────────────────────────

class TestWERComputation:
    """Unit tests for WER computation utility."""

    def test_perfect_match_wer(self):
        assert compute_wer("xin chào", "xin chào") == 0.0

    def test_one_substitution(self):
        wer = compute_wer("xin chào bạn", "xin chào tôi")
        assert abs(wer - 1 / 3) < 0.001

    def test_complete_miss(self):
        wer = compute_wer("xin chào", "tạm biệt")
        assert wer == 1.0

    def test_cer_perfect(self):
        assert compute_cer("xin chào", "xin chào") == 0.0

    def test_cer_one_char(self):
        cer = compute_cer("abc", "adc")
        assert abs(cer - 1 / 3) < 0.001


# ── KPI Targets ───────────────────────────────────────────────────────────────

class TestASRKPITargets:
    """Verify ASR benchmark results meet project KPIs."""

    WER_TARGET = 0.08   # ≤ 8%
    CER_TARGET = 0.04   # ≤ 4%

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("TEST_RESULTS_JSON", "test_results.json")),
        reason="No benchmark results file found. Run benchmarks first.",
    )
    def test_wer_meets_target(self):
        """WER must be ≤ 8% on ViMD test set."""
        with open(os.getenv("TEST_RESULTS_JSON", "test_results.json")) as f:
            results = json.load(f)
        wer = results.get("wer", 1.0)
        assert wer <= self.WER_TARGET, (
            f"WER={wer:.2%} exceeds target of {self.WER_TARGET:.2%}"
        )

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("TEST_RESULTS_JSON", "test_results.json")),
        reason="No benchmark results file found.",
    )
    def test_cer_meets_target(self):
        """CER must be ≤ 4% on ViMD test set."""
        with open(os.getenv("TEST_RESULTS_JSON", "test_results.json")) as f:
            results = json.load(f)
        cer = results.get("cer", 1.0)
        assert cer <= self.CER_TARGET, (
            f"CER={cer:.2%} exceeds target of {self.CER_TARGET:.2%}"
        )


# ── Dialect Accuracy Tests ────────────────────────────────────────────────────

class TestDialectKPITargets:
    """Verify dialect identification KPIs."""

    PROVINCE_TARGET = 0.70   # ≥ 70%
    REGIONAL_TARGET = 0.95   # ≥ 95%

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("TEST_RESULTS_JSON", "test_results.json")),
        reason="No benchmark results file found.",
    )
    def test_province_accuracy_meets_target(self):
        with open(os.getenv("TEST_RESULTS_JSON", "test_results.json")) as f:
            results = json.load(f)
        acc = results.get("province_accuracy", 0.0)
        assert acc >= self.PROVINCE_TARGET, (
            f"Province accuracy={acc:.2%} below target of {self.PROVINCE_TARGET:.2%}"
        )

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("TEST_RESULTS_JSON", "test_results.json")),
        reason="No benchmark results file found.",
    )
    def test_regional_accuracy_meets_target(self):
        with open(os.getenv("TEST_RESULTS_JSON", "test_results.json")) as f:
            results = json.load(f)
        acc = results.get("region_accuracy", 0.0)
        assert acc >= self.REGIONAL_TARGET, (
            f"Regional accuracy={acc:.2%} below target of {self.REGIONAL_TARGET:.2%}"
        )
