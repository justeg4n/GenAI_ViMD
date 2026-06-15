"""
Audio Denoiser
==============
Applies speech enhancement / noise reduction using:
  1. RNNoise (primary) — real-time neural denoiser
  2. noisereduce (secondary) — spectral gating
  3. wiener filter (fallback) — classic signal processing

Used in the ASR pipeline between VAD and Whisper.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioDenoiser:
    """
    Speech enhancement pipeline.
    Tries backends in order: RNNoise → noisereduce → wiener.

    Usage:
        denoiser = AudioDenoiser()
        clean_audio = denoiser.denoise(noisy_audio, sample_rate=16000)
    """

    TARGET_SR = 16_000

    def __init__(
        self,
        method: str = "auto",          # auto | rnnoise | noisereduce | wiener
        stationary: bool = False,       # noisereduce: True=stationary noise model
        prop_decrease: float = 0.8,     # noisereduce: proportion of noise to remove
        wiener_noise_std: float = 0.01, # wiener: assumed noise std dev
    ):
        self.method = method
        self.stationary = stationary
        self.prop_decrease = prop_decrease
        self.wiener_noise_std = wiener_noise_std

        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Auto-detect the best available denoising backend."""
        if self.method != "auto":
            return self.method
        try:
            import rnnoise
            return "rnnoise"
        except ImportError:
            pass
        try:
            import noisereduce
            return "noisereduce"
        except ImportError:
            pass
        return "wiener"

    def denoise(self, audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        """
        Denoise a 1D float32 audio array.

        Args:
            audio: float32 mono waveform
            sample_rate: audio sample rate

        Returns:
            Denoised float32 waveform (same length)
        """
        if self._backend == "rnnoise":
            return self._rnnoise_denoise(audio, sample_rate)
        elif self._backend == "noisereduce":
            return self._noisereduce_denoise(audio, sample_rate)
        else:
            return self._wiener_denoise(audio)

    def denoise_batch(
        self, audios: list[np.ndarray], sample_rate: int = 16_000
    ) -> list[np.ndarray]:
        """Denoise a batch of audio arrays."""
        return [self.denoise(a, sample_rate) for a in audios]

    # ── RNNoise ───────────────────────────────────────────────────────────────

    def _rnnoise_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Use RNNoise neural denoiser (real-time capable)."""
        try:
            import rnnoise
            # RNNoise expects 16kHz int16 frames of 480 samples
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    pass

            # Convert to int16
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            denoised = rnnoise.reduce_noise(pcm)
            return (denoised.astype(np.float32) / 32767.0)
        except Exception as e:
            logger.debug(f"RNNoise failed: {e}, falling back to noisereduce")
            return self._noisereduce_denoise(audio, sr)

    # ── noisereduce ───────────────────────────────────────────────────────────

    def _noisereduce_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Use noisereduce spectral gating."""
        try:
            import noisereduce as nr
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
            )
            return denoised.astype(np.float32)
        except Exception as e:
            logger.debug(f"noisereduce failed: {e}, falling back to wiener")
            return self._wiener_denoise(audio)

    # ── Wiener Filter ─────────────────────────────────────────────────────────

    def _wiener_denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple Wiener filter in the frequency domain.
        Assumes stationary Gaussian noise.
        """
        n = len(audio)
        spectrum = np.fft.rfft(audio)
        power = np.abs(spectrum) ** 2
        noise_power = self.wiener_noise_std ** 2 * n

        # Wiener gain: H(f) = |S(f)|² / (|S(f)|² + noise_power)
        gain = power / (power + noise_power)
        denoised_spectrum = gain * spectrum
        denoised = np.fft.irfft(denoised_spectrum, n=n)
        return denoised.astype(np.float32)

    @property
    def backend(self) -> str:
        return self._backend
