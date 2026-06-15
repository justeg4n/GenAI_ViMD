"""
VAD — Voice Activity Detection
================================
Wraps Silero VAD for reliable speech/silence segmentation.
Used as the first stage in the ASR pipeline before denoising.

Outputs timestamped speech segments for streaming inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment with timestamps."""
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int
    audio: np.ndarray
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class VAD:
    """
    Voice Activity Detector using Silero VAD.

    Silero VAD is a pre-trained, production-grade VAD model
    that runs efficiently on CPU and GPU.

    Usage:
        vad = VAD()
        segments = vad.detect(audio_array, sample_rate=16000)
        for seg in segments:
            print(f"{seg.start_s:.2f}s - {seg.end_s:.2f}s")
    """

    SAMPLE_RATE = 16_000
    SILERO_MODEL = "snakers4/silero-vad"

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        device: str = "auto",
        use_webrtc_fallback: bool = True,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.use_webrtc_fallback = use_webrtc_fallback

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._utils = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Silero VAD model."""
        try:
            model, utils = torch.hub.load(
                repo_or_dir=self.SILERO_MODEL,
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model.to(self.device)
            self._utils = utils
            logger.info(f"Silero VAD loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            if self.use_webrtc_fallback:
                logger.info("Falling back to WebRTC VAD")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        return_seconds: bool = True,
    ) -> list[SpeechSegment]:
        """
        Detect speech segments in an audio array.

        Returns:
            List of SpeechSegment objects with start/end timestamps.
        """
        if sample_rate != self.SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE)
                sample_rate = self.SAMPLE_RATE
            except ImportError:
                logger.warning("librosa not available for resampling")

        if self._model is not None and self._utils is not None:
            return self._silero_detect(audio, sample_rate)
        elif self.use_webrtc_fallback:
            return self._webrtc_detect(audio, sample_rate)
        else:
            # No VAD: treat entire audio as speech
            return [SpeechSegment(
                start_s=0.0,
                end_s=len(audio) / sample_rate,
                start_sample=0,
                end_sample=len(audio),
                audio=audio,
                confidence=1.0,
            )]

    def stream(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        sample_rate: int = 16_000,
        chunk_size_ms: int = 100,
    ) -> Generator[SpeechSegment, None, None]:
        """
        Stream VAD detection for real-time use.
        Buffers audio and yields complete speech segments.
        """
        buffer = np.array([], dtype=np.float32)
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)

        for chunk in audio_generator:
            buffer = np.concatenate([buffer, chunk])
            if len(buffer) >= chunk_samples:
                segments = self.detect(buffer, sample_rate)
                for seg in segments:
                    yield seg
                # Keep a small overlap to avoid cutting speech
                overlap = int(sample_rate * 0.1)
                buffer = buffer[-overlap:]

    def is_speech(
        self, audio: np.ndarray, sample_rate: int = 16_000
    ) -> Tuple[bool, float]:
        """
        Quick check if an audio chunk contains speech.
        Returns (is_speech, confidence).
        """
        segments = self.detect(audio, sample_rate)
        if not segments:
            return False, 0.0
        speech_duration = sum(s.duration_s for s in segments)
        total_duration = len(audio) / sample_rate
        confidence = speech_duration / max(total_duration, 1e-6)
        return confidence > 0.3, confidence

    # ── Silero Backend ────────────────────────────────────────────────────────

    def _silero_detect(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[SpeechSegment]:
        """Run Silero VAD detection."""
        try:
            get_speech_ts = self._utils[0]
            audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
            timestamps = get_speech_ts(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=self.window_size_samples,
                speech_pad_ms=self.speech_pad_ms,
            )
            segments = []
            for ts in timestamps:
                start = ts["start"]
                end = ts["end"]
                seg_audio = audio[start:end]
                segments.append(SpeechSegment(
                    start_s=start / sample_rate,
                    end_s=end / sample_rate,
                    start_sample=start,
                    end_sample=end,
                    audio=seg_audio,
                    confidence=1.0,
                ))
            return segments
        except Exception as e:
            logger.warning(f"Silero detection failed: {e}")
            return self._webrtc_detect(audio, sample_rate)

    # ── WebRTC Fallback ───────────────────────────────────────────────────────

    def _webrtc_detect(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[SpeechSegment]:
        """Fallback VAD using WebRTC."""
        try:
            import webrtcvad
            vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
            frame_ms = 30
            frame_samples = int(sample_rate * frame_ms / 1000)
            pcm = (audio * 32767).astype(np.int16).tobytes()

            speech_frames = []
            for i in range(0, len(audio), frame_samples):
                frame = pcm[i * 2:(i + frame_samples) * 2]
                if len(frame) < frame_samples * 2:
                    break
                is_speech = vad.is_speech(frame, sample_rate)
                speech_frames.append((i, i + frame_samples, is_speech))

            # Merge consecutive speech frames
            segments = []
            in_speech = False
            seg_start = 0
            for start, end, is_speech_frame in speech_frames:
                if is_speech_frame and not in_speech:
                    seg_start = start
                    in_speech = True
                elif not is_speech_frame and in_speech:
                    seg_audio = audio[seg_start:start]
                    if len(seg_audio) > 0:
                        segments.append(SpeechSegment(
                            start_s=seg_start / sample_rate,
                            end_s=start / sample_rate,
                            start_sample=seg_start,
                            end_sample=start,
                            audio=seg_audio,
                            confidence=0.8,
                        ))
                    in_speech = False
            return segments
        except ImportError:
            logger.warning("webrtcvad not available — no VAD fallback")
            return [SpeechSegment(
                start_s=0.0,
                end_s=len(audio) / sample_rate,
                start_sample=0,
                end_sample=len(audio),
                audio=audio,
                confidence=0.5,
            )]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def merge_segments(
        segments: list[SpeechSegment],
        max_gap_s: float = 0.3,
    ) -> list[SpeechSegment]:
        """Merge speech segments separated by less than max_gap_s."""
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.start_s - prev.end_s < max_gap_s:
                # Merge
                combined_audio = np.concatenate([prev.audio, seg.audio])
                merged[-1] = SpeechSegment(
                    start_s=prev.start_s,
                    end_s=seg.end_s,
                    start_sample=prev.start_sample,
                    end_sample=seg.end_sample,
                    audio=combined_audio,
                    confidence=max(prev.confidence, seg.confidence),
                )
            else:
                merged.append(seg)
        return merged
