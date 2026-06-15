"""
Feature Extractor
=================
Extracts acoustic features from audio for training:
  - Mel Spectrogram (80/128 bins)
  - MFCC (13/40 coefficients)
  - F0 / Pitch (RAPT + PyWorld)
  - Energy (RMS per frame)
  - Speaker Embeddings (SpeechBrain ECAPA-TDNN)
  - Prosody features (pitch contour, energy contour, duration)

Stores outputs in Parquet and Arrow formats for efficient downstream loading.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

logger = logging.getLogger(__name__)

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available")

try:
    import pyworld as pw
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False
    logger.warning("pyworld not available — F0 via parselmouth only")

try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

try:
    from speechbrain.pretrained import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False
    logger.warning("SpeechBrain not available — speaker embeddings skipped")


class FeatureExtractor:
    """
    Extracts and stores acoustic features for ViSpeech training.

    Usage:
        extractor = FeatureExtractor(output_dir="storage/datasets/features")
        features = extractor.extract(audio_array, sample_rate, metadata)
        extractor.save_batch(batch_features, "train_features.parquet")
    """

    DEFAULT_SR = 16_000
    N_MELS = 80
    N_MFCC = 40
    HOP_LENGTH = 160          # 10ms at 16kHz
    WIN_LENGTH = 400           # 25ms at 16kHz
    N_FFT = 512
    F_MIN = 80.0
    F_MAX = 7_600.0

    def __init__(
        self,
        output_dir: str = "storage/datasets/features",
        sample_rate: int = DEFAULT_SR,
        compute_speaker_emb: bool = True,
        device: str = "auto",
        n_mels: int = N_MELS,
        n_mfcc: int = N_MFCC,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.compute_speaker_emb = compute_speaker_emb
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._speaker_model = None
        if compute_speaker_emb and HAS_SPEECHBRAIN:
            self._load_speaker_model()

    # ── Speaker Model ─────────────────────────────────────────────────────────

    def _load_speaker_model(self) -> None:
        """Load ECAPA-TDNN speaker encoder from SpeechBrain."""
        try:
            self._speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.output_dir.parent / "hf_cache" / "speechbrain_spk"),
                run_opts={"device": self.device},
            )
            logger.info("SpeechBrain speaker encoder loaded")
        except Exception as e:
            logger.warning(f"Failed to load speaker model: {e}")
            self._speaker_model = None

    # ── Core Feature Extraction ───────────────────────────────────────────────

    def extract(
        self,
        audio: np.ndarray,
        sr: int,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Extract all features from a single audio array.

        Returns:
            dict with keys: mel, mfcc, f0, energy, pitch_mean, pitch_std,
                            energy_mean, energy_std, speaker_embedding, duration
        """
        if not HAS_LIBROSA:
            raise RuntimeError("librosa is required for feature extraction")

        audio = audio.astype(np.float32)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        features: dict = {}
        meta = metadata or {}

        # Duration
        features["duration"] = len(audio) / sr

        # ── Mel Spectrogram ──
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
            fmin=self.F_MIN,
            fmax=self.F_MAX,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features["mel"] = mel_db.T.astype(np.float32)           # (T, n_mels)
        features["mel_mean"] = float(mel_db.mean())
        features["mel_std"] = float(mel_db.std())

        # ── MFCC ──
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH,
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features["mfcc"] = mfcc.T.astype(np.float32)            # (T, n_mfcc)
        features["mfcc_delta"] = delta_mfcc.T.astype(np.float32)
        features["mfcc_delta2"] = delta2_mfcc.T.astype(np.float32)

        # ── Energy (RMS) ──
        rms = librosa.feature.rms(
            y=audio, frame_length=self.WIN_LENGTH, hop_length=self.HOP_LENGTH
        )[0]
        features["energy"] = rms.astype(np.float32)             # (T,)
        features["energy_mean"] = float(rms.mean())
        features["energy_std"] = float(rms.std())
        features["energy_max"] = float(rms.max())

        # ── F0 / Pitch ──
        f0, voiced = self._extract_f0(audio, sr)
        features["f0"] = f0.astype(np.float32)                  # (T,)
        features["voiced"] = voiced.astype(np.float32)
        voiced_f0 = f0[voiced > 0.5]
        features["pitch_mean"] = float(voiced_f0.mean()) if len(voiced_f0) > 0 else 0.0
        features["pitch_std"] = float(voiced_f0.std()) if len(voiced_f0) > 0 else 0.0
        features["pitch_min"] = float(voiced_f0.min()) if len(voiced_f0) > 0 else 0.0
        features["pitch_max"] = float(voiced_f0.max()) if len(voiced_f0) > 0 else 0.0
        features["voiced_ratio"] = float(voiced.mean())

        # ── Spectral Features ──
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.WIN_LENGTH, hop_length=self.HOP_LENGTH
        )[0]
        features["zcr_mean"] = float(zcr.mean())
        features["zcr_std"] = float(zcr.std())

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )[0]
        features["spectral_centroid_mean"] = float(spectral_centroid.mean())

        # ── Speaker Embedding ──
        if self.compute_speaker_emb and self._speaker_model is not None:
            features["speaker_embedding"] = self._extract_speaker_embedding(audio, sr)
        else:
            features["speaker_embedding"] = np.zeros(192, dtype=np.float32)

        # ── Metadata passthrough ──
        features.update({
            "filename": meta.get("filename", ""),
            "transcript": meta.get("transcript", ""),
            "speaker": meta.get("speaker", ""),
            "gender": int(meta.get("gender", -1)),
            "province": meta.get("province", ""),
            "region": meta.get("region", ""),
            "ethnicity": meta.get("ethnicity", "viet"),
            "language": meta.get("language", "vi"),
        })

        return features

    def _extract_f0(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 pitch contour using PyWorld (preferred) or librosa PYIN."""
        n_frames = 1 + len(audio) // self.HOP_LENGTH

        if HAS_PYWORLD:
            try:
                audio_d = audio.astype(np.float64)
                f0, timeaxis = pw.dio(audio_d, sr, frame_period=self.HOP_LENGTH / sr * 1000)
                f0 = pw.stonemask(audio_d, f0, timeaxis, sr)
                voiced = (f0 > 0).astype(np.float32)
                # Resize to match mel frames
                if len(f0) != n_frames:
                    from scipy.signal import resample
                    f0 = resample(f0, n_frames)
                    voiced = resample(voiced, n_frames)
                return f0.astype(np.float32), voiced.astype(np.float32)
            except Exception as e:
                logger.debug(f"PyWorld F0 failed: {e}, falling back to PYIN")

        # Fallback: librosa PYIN
        if HAS_LIBROSA:
            try:
                f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
                    audio, fmin=60, fmax=500, sr=sr,
                    hop_length=self.HOP_LENGTH, fill_na=0.0,
                )
                voiced = voiced_prob.astype(np.float32)
                return f0_pyin.astype(np.float32), voiced
            except Exception as e:
                logger.debug(f"PYIN F0 failed: {e}")

        return np.zeros(n_frames, dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract 192-dim ECAPA-TDNN speaker embedding."""
        try:
            tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                emb = self._speaker_model.encode_batch(tensor)
            return emb.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.debug(f"Speaker embedding failed: {e}")
            return np.zeros(192, dtype=np.float32)

    # ── Batch Processing ──────────────────────────────────────────────────────

    def extract_batch(
        self,
        samples: list[dict],
        audio_key: str = "audio",
        sr_key: str = "sampling_rate",
    ) -> list[dict]:
        """Extract features from a batch of dataset samples."""
        results = []
        for i, sample in enumerate(samples):
            audio_field = sample[audio_key]
            if isinstance(audio_field, dict):
                arr = np.array(audio_field["array"], dtype=np.float32)
                sr = audio_field["sampling_rate"]
            elif isinstance(audio_field, np.ndarray):
                arr = audio_field
                sr = sample.get(sr_key, self.sample_rate)
            else:
                logger.warning(f"Skipping sample {i}: unknown audio type")
                continue

            try:
                feats = self.extract(arr, sr, metadata=sample)
                results.append(feats)
            except Exception as e:
                logger.warning(f"Feature extraction failed for sample {i}: {e}")

        return results

    # ── Storage ───────────────────────────────────────────────────────────────

    def save_batch(
        self,
        features: list[dict],
        filename: str,
        compress: bool = True,
    ) -> Path:
        """
        Save extracted features to a Parquet file.
        Arrays (mel, mfcc, f0, etc.) are stored as list columns.
        """
        output_path = self.output_dir / filename
        records = []
        for f in features:
            record = {}
            for k, v in f.items():
                if isinstance(v, np.ndarray):
                    record[k] = v.tolist()    # Parquet list column
                else:
                    record[k] = v
            records.append(record)

        df = pd.DataFrame(records)
        compression = "snappy" if compress else None
        df.to_parquet(str(output_path), index=False, compression=compression)
        logger.info(f"Saved {len(records):,} feature records to {output_path}")
        return output_path

    def save_arrow(self, features: list[dict], filename: str) -> Path:
        """Save features to Apache Arrow IPC format for fast random access."""
        output_path = self.output_dir / filename
        records = []
        for f in features:
            record = {}
            for k, v in f.items():
                if isinstance(v, np.ndarray):
                    record[k] = v.tolist()
                else:
                    record[k] = v
            records.append(record)

        table = pa.Table.from_pylist(records)
        with pa.ipc.new_file(str(output_path), table.schema) as writer:
            writer.write_table(table)
        logger.info(f"Saved {len(records):,} feature records to {output_path}")
        return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Feature Extractor CLI")
    parser.add_argument("--data-dir", required=True, help="Path to ViMD parquet directory")
    parser.add_argument("--output-dir", default="storage/datasets/features")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--no-speaker-emb", action="store_true")
    args = parser.parse_args()

    # Import here to avoid circular dependency
    from dataset.ingestion.vimd_loader import ViMDLoader

    loader = ViMDLoader(data_dir=args.data_dir, max_samples=args.max_samples)
    extractor = FeatureExtractor(
        output_dir=args.output_dir,
        compute_speaker_emb=not args.no_speaker_emb,
    )

    samples = list(loader.stream_split(args.split))
    print(f"Processing {len(samples)} samples from split '{args.split}'...")

    batch = []
    for audio, sr, meta in samples:
        batch.append({"audio": {"array": audio, "sampling_rate": sr}, **meta})

    features = extractor.extract_batch(batch)
    extractor.save_batch(features, f"{args.split}_features.parquet")
    print(f"Done! Features saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
