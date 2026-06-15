"""
Unit Tests — Audio Validation
"""

import numpy as np
import pytest


class TestAudioValidator:
    """Unit tests for AudioValidator."""

    @pytest.fixture
    def validator(self):
        from dataset.validation.audio_validator import AudioValidator
        return AudioValidator()

    def _make_audio(self, duration_s=2.0, sr=16000, amplitude=0.5):
        """Generate synthetic speech-like audio."""
        t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
        audio = amplitude * np.sin(2 * np.pi * 200 * t)  # 200Hz tone
        audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)
        return audio

    def test_valid_audio_passes(self, validator):
        audio = self._make_audio(duration_s=3.0)
        result = validator.validate_array(audio, sr=16000, filename="test.wav")
        assert result.sample_rate == 16000
        assert result.duration_s == pytest.approx(3.0, rel=0.01)
        # Valid audio should not have clipping errors
        assert not any("Clipping" in e for e in result.errors)

    def test_too_short_audio_invalid(self, validator):
        audio = self._make_audio(duration_s=0.1)
        result = validator.validate_array(audio, sr=16000)
        assert not result.valid
        assert any("min" in e.lower() for e in result.errors)

    def test_too_long_audio_invalid(self, validator):
        audio = self._make_audio(duration_s=35.0)
        result = validator.validate_array(audio, sr=16000)
        assert not result.valid
        assert any("max" in e.lower() for e in result.errors)

    def test_clipping_detection(self, validator):
        audio = self._make_audio(duration_s=2.0, amplitude=1.5)
        audio = np.clip(audio, -1.0, 1.0)
        # Force lots of clipping
        audio[:500] = 0.999
        result = validator.validate_array(audio, sr=16000)
        assert any("Clip" in e for e in result.errors)
        assert not result.valid

    def test_normalize_lufs(self):
        from dataset.validation.audio_validator import AudioValidator
        audio = np.random.randn(32000).astype(np.float32) * 0.5
        normalized = AudioValidator.normalize_lufs(audio, 16000, target_lufs=-23.0)
        assert len(normalized) == len(audio)
        assert normalized.dtype == np.float32

    def test_remove_dc_offset(self):
        from dataset.validation.audio_validator import AudioValidator
        audio = np.ones(16000, dtype=np.float32) * 0.1  # High DC offset
        audio += np.random.randn(16000).astype(np.float32) * 0.01
        cleaned = AudioValidator.remove_dc_offset(audio)
        assert abs(np.mean(cleaned)) < 0.01

    def test_batch_validation(self, validator):
        samples = [
            {"audio": {"array": self._make_audio(2.0), "sampling_rate": 16000}, "filename": "a"},
            {"audio": {"array": self._make_audio(0.1), "sampling_rate": 16000}, "filename": "b"},
        ]
        results = validator.validate_batch(samples)
        assert len(results) == 2
        assert results[0].valid  # 2s audio is valid
        assert not results[1].valid  # 0.1s is too short


class TestLabelValidator:
    """Unit tests for LabelValidator."""

    @pytest.fixture
    def validator(self):
        from dataset.validation.label_validator import LabelValidator
        return LabelValidator()

    def test_valid_row(self, validator):
        row = {
            "text": "Đây là bài kiểm tra giọng địa phương",
            "province": "HN",
            "region": "North",
            "gender": 0,
            "length": 5.0,
        }
        is_valid, errors = validator.validate_single(row)
        assert is_valid, f"Expected valid, got errors: {errors}"

    def test_empty_transcript_invalid(self, validator):
        row = {"text": "", "province": "HN", "region": "North", "gender": 0, "length": 3.0}
        is_valid, errors = validator.validate_single(row)
        assert not is_valid

    def test_invalid_region(self, validator):
        row = {"text": "Test text here", "province": "HN", "region": "Unknown", "gender": 0, "length": 3.0}
        is_valid, errors = validator.validate_single(row)
        assert not is_valid
        assert any("region" in e.lower() for e in errors)

    def test_invalid_gender(self, validator):
        row = {"text": "Test text here", "province": "HN", "region": "North", "gender": 5, "length": 3.0}
        is_valid, errors = validator.validate_single(row)
        assert not is_valid

    def test_dataframe_validation_removes_duplicates(self, validator):
        import pandas as pd
        df = pd.DataFrame([
            {"text": "Xin chào", "province": "HN", "region": "North", "gender": 0, "length": 3.0,
             "filename": "a", "speakerID": "spk_1"},
            {"text": "Xin chào", "province": "HN", "region": "North", "gender": 0, "length": 3.0,
             "filename": "a", "speakerID": "spk_1"},  # duplicate
        ])
        clean_df, report = validator.validate_dataframe(df)
        assert report.removed_duplicates == 1
        assert len(clean_df) == 1
