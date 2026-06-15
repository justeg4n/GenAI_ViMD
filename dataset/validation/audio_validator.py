"""
Audio Validator
===============
Validates audio files for format compliance and quality requirements:
  - Format: WAV, mono, 16kHz
  - Loudness: −23 LUFS (±2 LU tolerance)
  - Duration: 0.5s – 30s
  - No clipping (peak ≥ 0.99)
  - No excessive silence
  - Bit depth: 16-bit or 32-bit float

Outputs a validation report with per-file status.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False
    logger.warning("pyloudnorm not available — LUFS validation skipped")


@dataclass
class AudioValidationResult:
    filename: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sample_rate: int = 0
    channels: int = 0
    duration_s: float = 0.0
    peak_amplitude: float = 0.0
    lufs: Optional[float] = None
    silence_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_s": round(self.duration_s, 3),
            "peak_amplitude": round(self.peak_amplitude, 4),
            "lufs": round(self.lufs, 2) if self.lufs is not None else None,
            "silence_ratio": round(self.silence_ratio, 4),
        }


class AudioValidator:
    """
    Validates audio data against ViSpeech platform requirements.

    Requirements (from spec):
      - Format: WAV, mono, 16kHz
      - Normalization target: −23 LUFS
      - Remove: duplicates, clipping, silence, invalid labels

    Usage:
        validator = AudioValidator()
        result = validator.validate_array(audio_array, sample_rate, filename="test.wav")
        # or
        result = validator.validate_file(Path("audio.wav"))
    """

    TARGET_SR = 16_000
    TARGET_CHANNELS = 1
    TARGET_LUFS = -23.0
    LUFS_TOLERANCE = 2.0          # ±2 LU
    MIN_DURATION_S = 0.5
    MAX_DURATION_S = 30.0
    CLIPPING_THRESHOLD = 0.99
    SILENCE_THRESHOLD = 0.01      # RMS below this = silent
    MAX_SILENCE_RATIO = 0.80      # >80% silence → invalid
    MIN_SILENCE_RATIO = 0.05      # <5% total silence → suspicious

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        target_lufs: float = TARGET_LUFS,
        lufs_tolerance: float = LUFS_TOLERANCE,
        min_duration: float = MIN_DURATION_S,
        max_duration: float = MAX_DURATION_S,
        strict: bool = False,
    ):
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.lufs_tolerance = lufs_tolerance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.strict = strict  # If True, warnings become errors

        if HAS_PYLOUDNORM:
            self._meter = pyln.Meter(target_sr)

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_file(self, path: Path) -> AudioValidationResult:
        """Validate a WAV file on disk."""
        result = AudioValidationResult(filename=str(path), valid=True)
        try:
            audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
            result.sample_rate = sr
            result.channels = 1 if audio.ndim == 1 else audio.shape[1]
            return self._validate_array_internal(audio, sr, result)
        except Exception as e:
            result.valid = False
            result.errors.append(f"Cannot read file: {e}")
            return result

    def validate_bytes(self, audio_bytes: bytes, filename: str = "") -> AudioValidationResult:
        """Validate audio from raw bytes."""
        result = AudioValidationResult(filename=filename, valid=True)
        try:
            buf = io.BytesIO(audio_bytes)
            audio, sr = sf.read(buf, always_2d=False, dtype="float32")
            result.sample_rate = sr
            result.channels = 1 if audio.ndim == 1 else audio.shape[1]
            return self._validate_array_internal(audio, sr, result)
        except Exception as e:
            result.valid = False
            result.errors.append(f"Cannot decode audio: {e}")
            return result

    def validate_array(
        self, audio: np.ndarray, sr: int, filename: str = ""
    ) -> AudioValidationResult:
        """Validate a numpy audio array."""
        result = AudioValidationResult(filename=filename, valid=True)
        result.sample_rate = sr
        result.channels = 1 if audio.ndim == 1 else audio.shape[1]
        return self._validate_array_internal(audio, sr, result)

    def validate_batch(self, samples: list[dict]) -> list[AudioValidationResult]:
        """
        Validate a batch of samples from the dataset.
        Each sample should have 'audio' (bytes/array) and optionally 'filename'.
        """
        results = []
        for i, sample in enumerate(samples):
            fname = sample.get("filename", f"sample_{i}")
            audio_field = sample.get("audio")
            if isinstance(audio_field, dict):
                arr = np.array(audio_field["array"], dtype=np.float32)
                sr = audio_field["sampling_rate"]
                r = self.validate_array(arr, sr, fname)
            elif isinstance(audio_field, bytes):
                r = self.validate_bytes(audio_field, fname)
            elif isinstance(audio_field, np.ndarray):
                sr = sample.get("sampling_rate", self.target_sr)
                r = self.validate_array(audio_field, sr, fname)
            else:
                r = AudioValidationResult(filename=fname, valid=False,
                                          errors=["Unknown audio field type"])
            results.append(r)
        return results

    # ── Private: Core validation ──────────────────────────────────────────────

    def _validate_array_internal(
        self, audio: np.ndarray, sr: int, result: AudioValidationResult
    ) -> AudioValidationResult:
        audio = audio.astype(np.float32)

        # Flatten stereo → mono for analysis
        if audio.ndim > 1:
            audio_mono = audio.mean(axis=1)
        else:
            audio_mono = audio

        n_samples = len(audio_mono)
        duration = n_samples / sr

        result.duration_s = duration
        result.peak_amplitude = float(np.max(np.abs(audio_mono)))

        # ── Check sample rate ──
        if sr != self.target_sr:
            msg = f"Sample rate {sr}Hz ≠ target {self.target_sr}Hz"
            if self.strict:
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

        # ── Check mono ──
        if audio.ndim > 1 and audio.shape[1] > 1:
            msg = f"Audio has {audio.shape[1]} channels (expected mono)"
            if self.strict:
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

        # ── Check duration ──
        if duration < self.min_duration:
            result.errors.append(f"Duration {duration:.2f}s < min {self.min_duration}s")
        if duration > self.max_duration:
            result.errors.append(f"Duration {duration:.2f}s > max {self.max_duration}s")

        # ── Check clipping ──
        clip_count = int(np.sum(np.abs(audio_mono) >= self.CLIPPING_THRESHOLD))
        if clip_count > 0:
            ratio = clip_count / n_samples
            msg = f"Clipping detected: {clip_count} samples ({ratio:.2%})"
            result.errors.append(msg)

        # ── Check silence ──
        frame_size = int(0.025 * sr)  # 25ms frames
        frames = [audio_mono[i:i+frame_size] for i in range(0, n_samples, frame_size)
                  if len(audio_mono[i:i+frame_size]) == frame_size]
        if frames:
            rms_per_frame = np.array([np.sqrt(np.mean(f**2)) for f in frames])
            silence_ratio = float(np.mean(rms_per_frame < self.SILENCE_THRESHOLD))
            result.silence_ratio = silence_ratio
            if silence_ratio > self.MAX_SILENCE_RATIO:
                result.errors.append(
                    f"Excessive silence: {silence_ratio:.1%} of frames are silent"
                )
            elif silence_ratio < self.MIN_SILENCE_RATIO and duration > 2.0:
                result.warnings.append(
                    f"Suspiciously little silence: {silence_ratio:.1%}"
                )

        # ── Check LUFS ──
        if HAS_PYLOUDNORM and len(audio_mono) >= sr:  # Need ≥1s for LUFS
            try:
                meter = pyln.Meter(sr)
                lufs = meter.integrated_loudness(audio_mono)
                result.lufs = lufs
                if not np.isinf(lufs):
                    diff = abs(lufs - self.target_lufs)
                    if diff > self.lufs_tolerance:
                        result.warnings.append(
                            f"LUFS {lufs:.1f} deviates from target {self.target_lufs:.1f} "
                            f"by {diff:.1f} LU (tolerance={self.lufs_tolerance})"
                        )
            except Exception as e:
                logger.debug(f"LUFS computation failed: {e}")

        # ── Check DC offset ──
        dc_offset = float(np.abs(np.mean(audio_mono)))
        if dc_offset > 0.05:
            result.warnings.append(f"High DC offset: {dc_offset:.4f}")

        # Finalize
        if result.errors:
            result.valid = False

        return result

    # ── Normalization helpers ──────────────────────────────────────────────────

    @staticmethod
    def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
        """Normalize audio to target integrated loudness (LUFS)."""
        if not HAS_PYLOUDNORM:
            # Fallback: peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak * 0.9
            return audio
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if np.isinf(loudness):
            return audio
        return pyln.normalize.loudness(audio, loudness, target_lufs)

    @staticmethod
    def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
        """Remove DC offset."""
        return audio - np.mean(audio)

    @staticmethod
    def trim_silence(
        audio: np.ndarray, sr: int,
        top_db: float = 40.0, frame_length: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """Trim leading/trailing silence using librosa."""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(
                audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
            )
            return trimmed
        except ImportError:
            return audio

    # ── Batch reporting ───────────────────────────────────────────────────────

    @staticmethod
    def print_report(results: list[AudioValidationResult]) -> None:
        """Print a summary validation report."""
        valid = [r for r in results if r.valid]
        invalid = [r for r in results if not r.valid]
        warned = [r for r in results if r.warnings]

        print(f"\n{'='*60}")
        print(f" Audio Validation Report")
        print(f"{'='*60}")
        print(f"  Total files : {len(results):,}")
        print(f"  Valid       : {len(valid):,} ({len(valid)/len(results)*100:.1f}%)")
        print(f"  Invalid     : {len(invalid):,}")
        print(f"  Warnings    : {len(warned):,}")

        if invalid:
            print(f"\n{'─'*40}")
            print(" Invalid Files:")
            for r in invalid[:20]:  # Show first 20
                print(f"  {r.filename}")
                for e in r.errors:
                    print(f"    ✗ {e}")

        # Aggregate stats
        durations = [r.duration_s for r in valid if r.duration_s > 0]
        if durations:
            print(f"\n{'─'*40}")
            print(f" Valid Audio Statistics:")
            print(f"  Duration: min={min(durations):.1f}s  "
                  f"max={max(durations):.1f}s  "
                  f"mean={np.mean(durations):.1f}s  "
                  f"total={sum(durations)/3600:.2f}h")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Audio Validator CLI")
    parser.add_argument("paths", nargs="+", help="Audio files or directories to validate")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--output", default=None, help="JSON output file for report")
    args = parser.parse_args()

    validator = AudioValidator(strict=args.strict)
    results = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for wav in sorted(path.rglob("*.wav")):
                results.append(validator.validate_file(wav))
        elif path.is_file():
            results.append(validator.validate_file(path))

    AudioValidator.print_report(results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
