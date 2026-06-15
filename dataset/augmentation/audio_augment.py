"""
Audio Augmentation Pipeline
============================
Implements all augmentation strategies required by the spec:
  - SpecAugment (time + frequency masking)
  - Noise Injection (white, pink, environmental)
  - Room Simulation / Reverberation (RIR convolution)
  - Pitch Shift
  - Speed Shift (tempo change)
  - Codec Simulation (MP3/OGG compression artifacts)
  - Mixup (alpha blending of two samples)
  - Frequency Masking
  - Time Masking

Designed for both offline augmentation (dataset pre-processing) and
online augmentation (during training DataLoader).
"""

from __future__ import annotations

import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import audiomentations as A
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    logger.warning("audiomentations not available — some augmentations use manual implementations")

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# ── SpecAugment (waveform-level via spectrogram manipulation) ──────────────────

class SpecAugment:
    """
    SpecAugment: Time and Frequency masking on mel spectrograms.
    Can be applied to mel features (numpy arrays) or integrated into training.

    Reference: Park et al. 2019 "SpecAugment"
    """

    def __init__(
        self,
        freq_mask_param: int = 27,      # Max freq bins to mask
        time_mask_param: int = 100,     # Max time frames to mask
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_value: float = 0.0,   # Fill value (0 or mean)
        time_warp: bool = False,        # Enable time warping
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_value = freq_mask_value
        self.time_warp = time_warp

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to a mel spectrogram.
        Args:
            mel: (T, F) array
        Returns:
            Augmented mel: (T, F)
        """
        mel = mel.copy()
        T, F = mel.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, F))
            f0 = random.randint(0, F - f)
            mel[:, f0:f0 + f] = self.freq_mask_value

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T))
            t0 = random.randint(0, T - t)
            mel[t0:t0 + t, :] = self.freq_mask_value

        return mel

    def apply_torch(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply to a PyTorch tensor (B, F, T) or (F, T)."""
        arr = mel.numpy() if mel.is_cpu else mel.cpu().numpy()
        if arr.ndim == 3:
            augmented = np.stack([self(m.T).T for m in arr])
        else:
            augmented = self(arr.T).T
        return torch.from_numpy(augmented).to(mel.device)


# ── Waveform Augmentations ────────────────────────────────────────────────────

class NoiseInjector:
    """Add white/pink/environmental noise at a specified SNR."""

    def __init__(
        self,
        snr_db_min: float = 5.0,
        snr_db_max: float = 30.0,
        noise_types: tuple = ("white", "pink"),
        noise_dir: Optional[str] = None,
    ):
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.noise_types = noise_types
        self.noise_files: list[Path] = []
        if noise_dir:
            self.noise_files = list(Path(noise_dir).rglob("*.wav"))

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        snr = random.uniform(self.snr_db_min, self.snr_db_max)
        noise_type = random.choice(self.noise_types)

        if noise_type == "file" and self.noise_files:
            noise = self._load_noise_file(sr, len(audio))
        elif noise_type == "pink":
            noise = self._pink_noise(len(audio))
        else:
            noise = np.random.randn(len(audio)).astype(np.float32)

        # Scale noise to target SNR
        signal_rms = np.sqrt(np.mean(audio ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms > 0:
            target_noise_rms = signal_rms / (10 ** (snr / 20))
            noise = noise * (target_noise_rms / noise_rms)

        return np.clip(audio + noise, -1.0, 1.0)

    def _pink_noise(self, n: int) -> np.ndarray:
        """Generate pink (1/f) noise via spectral shaping."""
        white = np.random.randn(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-6  # Avoid div-by-zero
        pink_fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=n)
        return pink.astype(np.float32)

    def _load_noise_file(self, sr: int, length: int) -> np.ndarray:
        if not HAS_SF:
            return np.random.randn(length).astype(np.float32)
        path = random.choice(self.noise_files)
        noise, noise_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if noise_sr != sr and HAS_LIBROSA:
            noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
        # Tile to match length
        if len(noise) < length:
            reps = (length // len(noise)) + 1
            noise = np.tile(noise, reps)
        start = random.randint(0, len(noise) - length)
        return noise[start:start + length].astype(np.float32)


class RoomSimulator:
    """Simulate room acoustics using Random Impulse Response (RIR) convolution."""

    def __init__(
        self,
        rir_dir: Optional[str] = None,
        reverb_prob: float = 0.5,
        rt60_min: float = 0.1,
        rt60_max: float = 1.5,
    ):
        self.reverb_prob = reverb_prob
        self.rt60_min = rt60_min
        self.rt60_max = rt60_max
        self.rir_files: list[Path] = []
        if rir_dir:
            self.rir_files = list(Path(rir_dir).rglob("*.wav"))

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.reverb_prob:
            return audio

        if self.rir_files and HAS_SF:
            return self._convolve_rir(audio, sr)
        return self._synthetic_reverb(audio, sr)

    def _convolve_rir(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Convolve with a real Room Impulse Response."""
        rir_path = random.choice(self.rir_files)
        rir, rir_sr = sf.read(str(rir_path), dtype="float32", always_2d=False)
        if rir.ndim > 1:
            rir = rir[:, 0]
        if rir_sr != sr and HAS_LIBROSA:
            rir = librosa.resample(rir, orig_sr=rir_sr, target_sr=sr)
        rir = rir / (np.abs(rir).max() + 1e-8)
        reverbed = np.convolve(audio, rir, mode="full")[:len(audio)]
        return np.clip(reverbed, -1.0, 1.0).astype(np.float32)

    def _synthetic_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Generate synthetic reverb via exponential decay convolution."""
        rt60 = random.uniform(self.rt60_min, self.rt60_max)
        decay_samples = int(rt60 * sr)
        t = np.arange(decay_samples) / sr
        rir = np.exp(-6.9 * t / rt60) * np.random.randn(decay_samples)
        rir = rir.astype(np.float32) / (np.abs(rir).max() + 1e-8)
        reverbed = np.convolve(audio, rir, mode="full")[:len(audio)]
        return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


class PitchShifter:
    """Shift pitch without changing speed."""

    def __init__(self, semitones_min: float = -3.0, semitones_max: float = 3.0):
        self.semitones_min = semitones_min
        self.semitones_max = semitones_max

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not HAS_LIBROSA:
            return audio
        n_steps = random.uniform(self.semitones_min, self.semitones_max)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


class SpeedShifter:
    """Change tempo without affecting pitch."""

    def __init__(self, rate_min: float = 0.85, rate_max: float = 1.15):
        self.rate_min = rate_min
        self.rate_max = rate_max

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not HAS_LIBROSA:
            return audio
        rate = random.uniform(self.rate_min, self.rate_max)
        return librosa.effects.time_stretch(audio, rate=rate)


class CodecSimulator:
    """Simulate codec degradation (MP3/OGG) artifacts."""

    def __init__(
        self,
        codecs: tuple = ("mp3", "ogg"),
        bitrates: tuple = (8, 16, 32, 64),
    ):
        self.codecs = codecs
        self.bitrates = bitrates

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if not HAS_SF:
            return audio
        codec = random.choice(self.codecs)
        bitrate = random.choice(self.bitrates)
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{codec}", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, sr)
            degraded, _ = sf.read(tmp_path, dtype="float32", always_2d=False)
            if len(degraded) != len(audio):
                degraded = degraded[:len(audio)] if len(degraded) > len(audio) else \
                           np.pad(degraded, (0, len(audio) - len(degraded)))
            os.unlink(tmp_path)
            return degraded
        except Exception as e:
            logger.debug(f"Codec simulation failed: {e}")
            return audio


class Mixup:
    """
    Mixup augmentation: blend two audio samples.
    Also blends labels proportionally (for classification tasks).
    """

    def __init__(self, alpha: float = 0.4, prob: float = 0.3):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        label1: Optional[np.ndarray] = None,
        label2: Optional[np.ndarray] = None,
    ) -> tuple:
        if random.random() > self.prob:
            return audio1, label1

        lam = np.random.beta(self.alpha, self.alpha)

        # Match lengths
        min_len = min(len(audio1), len(audio2))
        a1 = audio1[:min_len]
        a2 = audio2[:min_len]

        mixed_audio = lam * a1 + (1 - lam) * a2
        mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)

        if label1 is not None and label2 is not None:
            mixed_label = lam * label1 + (1 - lam) * label2
            return mixed_audio, mixed_label

        return mixed_audio, label1


# ── Unified Augmentation Pipeline ─────────────────────────────────────────────

class AudioAugmentationPipeline:
    """
    Combines all augmentation strategies into a configurable pipeline.

    Usage:
        pipeline = AudioAugmentationPipeline(p=0.5)
        augmented_audio = pipeline(audio_array, sample_rate=16000)
    """

    def __init__(
        self,
        p: float = 0.5,                   # Overall augmentation probability
        noise_p: float = 0.4,
        reverb_p: float = 0.3,
        pitch_p: float = 0.3,
        speed_p: float = 0.3,
        codec_p: float = 0.2,
        noise_dir: Optional[str] = None,
        rir_dir: Optional[str] = None,
        snr_min: float = 5.0,
        snr_max: float = 30.0,
        pitch_semitones: float = 3.0,
        speed_range: tuple = (0.85, 1.15),
    ):
        self.p = p
        self.noise_p = noise_p
        self.reverb_p = reverb_p
        self.pitch_p = pitch_p
        self.speed_p = speed_p
        self.codec_p = codec_p

        self.noise_injector = NoiseInjector(
            snr_db_min=snr_min, snr_db_max=snr_max, noise_dir=noise_dir
        )
        self.room_simulator = RoomSimulator(rir_dir=rir_dir)
        self.pitch_shifter = PitchShifter(
            semitones_min=-pitch_semitones, semitones_max=pitch_semitones
        )
        self.speed_shifter = SpeedShifter(rate_min=speed_range[0], rate_max=speed_range[1])
        self.codec_simulator = CodecSimulator()
        self.spec_augment = SpecAugment()
        self.mixup = Mixup()

    def __call__(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        """Apply random augmentations to a waveform."""
        if random.random() > self.p:
            return audio

        # Apply augmentations probabilistically
        if random.random() < self.reverb_p:
            audio = self.room_simulator(audio, sr)
        if random.random() < self.noise_p:
            audio = self.noise_injector(audio, sr)
        if random.random() < self.pitch_p:
            audio = self.pitch_shifter(audio, sr)
        if random.random() < self.speed_p:
            audio = self.speed_shifter(audio, sr)
        if random.random() < self.codec_p:
            audio = self.codec_simulator(audio, sr)

        return audio.astype(np.float32)

    def augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to a mel spectrogram array (T, F)."""
        return self.spec_augment(mel)

    def augment_dataset_sample(self, sample: dict) -> dict:
        """
        Augment a full dataset sample dict.
        Expects sample["audio"] = {"array": ..., "sampling_rate": ...}
        """
        audio_field = sample.get("audio", {})
        if isinstance(audio_field, dict):
            audio = np.array(audio_field["array"], dtype=np.float32)
            sr = audio_field["sampling_rate"]
        elif isinstance(audio_field, np.ndarray):
            audio = audio_field
            sr = sample.get("sampling_rate", 16_000)
        else:
            return sample

        augmented = self(audio, sr)
        result = sample.copy()
        if isinstance(audio_field, dict):
            result["audio"] = {**audio_field, "array": augmented}
        else:
            result["audio"] = augmented
        return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Audio Augmentation Pipeline CLI")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--reverb", action="store_true")
    parser.add_argument("--pitch", action="store_true")
    parser.add_argument("--speed", action="store_true")
    args = parser.parse_args()

    if not (HAS_SF and HAS_LIBROSA):
        print("Error: soundfile and librosa are required")
        return

    audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
    pipeline = AudioAugmentationPipeline(p=1.0,
        noise_p=1.0 if args.noise else 0.0,
        reverb_p=1.0 if args.reverb else 0.0,
        pitch_p=1.0 if args.pitch else 0.0,
        speed_p=1.0 if args.speed else 0.0,
    )
    augmented = pipeline(audio, sr)
    sf.write(args.output, augmented, sr)
    print(f"Saved augmented audio to {args.output}")


if __name__ == "__main__":
    main()
