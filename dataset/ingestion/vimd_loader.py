"""
ViMD Dataset Loader
===================
Loads the Vietnamese Multi-Dialect (ViMD) dataset from parquet shards.
Supports both local parquet files and HuggingFace Hub streaming.

Dataset attributes per spec:
  set, filename, text, length, province, region, speakerID, gender
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

# ── Province → Region mapping (63 provinces) ──────────────────────────────────
PROVINCE_TO_REGION: dict[str, str] = {
    # North (Bắc Bộ)
    "hanoi": "North", "hanam": "North", "hungyen": "North", "ninhbinh": "North",
    "namdinh": "North", "thaibinh": "North", "bacninh": "North", "bacgiang": "North",
    "haiduong": "North", "haiphong": "North", "quangninh": "North", "langson": "North",
    "caobang": "North", "backan": "North", "thainguyen": "North", "tuyenquang": "North",
    "hagiang": "North", "laocai": "North", "yenbai": "North", "phutho": "North",
    "vinhphuc": "North", "hoabinh": "North", "sonla": "North", "dienbien": "North",
    "laichau": "North",
    # Central (Trung Bộ)
    "thanhhoa": "Central", "nghean": "Central", "hatinh": "Central", "quangbinh": "Central",
    "quangtri": "Central", "thuathienhue": "Central", "danang": "Central",
    "quangnam": "Central", "quangngai": "Central", "binhdinh": "Central",
    "phuyen": "Central", "khanhhoa": "Central", "ninhthuan": "Central",
    "binhthuan": "Central", "kontum": "Central", "gialai": "Central",
    "daklak": "Central", "daknong": "Central", "lamdong": "Central",
    # South (Nam Bộ)
    "hochiminh": "South", "binhduong": "South", "dongnai": "South",
    "bariavungtau": "South", "longAn": "South", "tiengiang": "South",
    "bentre": "South", "travinh": "South", "vinhlong": "South",
    "dongthap": "South", "angiang": "South", "kiengiang": "South",
    "cantho": "South", "haugiang": "South", "soctrang": "South",
    "baclieu": "South", "camau": "South", "binhphuoc": "South", "tayninh": "South",
}


class ViMDLoader:
    """
    Loads ViMD dataset shards from local parquet files or HuggingFace Hub.

    Usage:
        loader = ViMDLoader(data_dir="/path/to/ViMD/data")
        dataset = loader.load_all()
        for sample in loader.stream_split("train"):
            audio_array, sample_rate, metadata = sample
    """

    SAMPLE_RATE = 16_000
    HF_REPO = "doof-ferb/ViMD"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        use_hf: bool = False,
        hf_token: Optional[str] = None,
        max_samples: Optional[int] = None,
        target_sr: int = 16_000,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_hf = use_hf
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.max_samples = max_samples
        self.target_sr = target_sr
        self._splits = ["train", "valid", "test"]

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> DatasetDict:
        """Load full DatasetDict (train/valid/test)."""
        if self.use_hf:
            return self._load_from_hub()
        return self._load_from_local()

    def load_split(self, split: str) -> Dataset:
        """Load a single split ('train', 'valid', 'test')."""
        assert split in self._splits, f"split must be one of {self._splits}"
        dd = self.load_all()
        return dd[split]

    def stream_split(self, split: str) -> Iterator[tuple[np.ndarray, int, dict]]:
        """
        Lazily stream (audio_array, sample_rate, metadata) tuples.
        Memory-efficient for large splits.
        """
        dataset = self.load_split(split)
        count = 0
        for row in dataset:
            if self.max_samples and count >= self.max_samples:
                break
            audio_array, sr = self._decode_audio(row["audio"])
            meta = self._build_metadata(row)
            yield audio_array, sr, meta
            count += 1

    def to_torch_dataset(self, split: str) -> "ViMDTorchDataset":
        """Wrap a split as a PyTorch Dataset."""
        return ViMDTorchDataset(self.load_split(split), target_sr=self.target_sr)

    # ── Private: Load ─────────────────────────────────────────────────────────

    def _load_from_hub(self) -> DatasetDict:
        logger.info(f"Loading ViMD from HuggingFace Hub: {self.HF_REPO}")
        return load_dataset(
            self.HF_REPO,
            token=self.hf_token,
            trust_remote_code=True,
        )

    def _load_from_local(self) -> DatasetDict:
        if self.data_dir is None:
            raise ValueError("data_dir must be set when use_hf=False")
        logger.info(f"Loading ViMD from local parquet: {self.data_dir}")
        splits_data: dict[str, Dataset] = {}
        for split in self._splits:
            shards = sorted(self.data_dir.glob(f"{split}-*.parquet"))
            if not shards:
                logger.warning(f"No parquet files found for split '{split}' in {self.data_dir}")
                continue
            dfs: list[pd.DataFrame] = []
            for shard in shards:
                logger.debug(f"  Loading shard: {shard.name}")
                df = pd.read_parquet(shard)
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            if self.max_samples:
                df_all = df_all.head(self.max_samples)
            splits_data[split] = Dataset.from_pandas(df_all)
            logger.info(f"  [{split}] {len(df_all):,} samples loaded")
        return DatasetDict(splits_data)

    # ── Private: Audio ────────────────────────────────────────────────────────

    def _decode_audio(self, audio_field) -> tuple[np.ndarray, int]:
        """
        Decode audio from HuggingFace audio dict or raw bytes.
        Returns (array float32 mono, sample_rate).
        """
        if isinstance(audio_field, dict):
            # HF format: {"array": np.ndarray, "sampling_rate": int, "path": str}
            arr = np.array(audio_field["array"], dtype=np.float32)
            sr = audio_field["sampling_rate"]
        elif isinstance(audio_field, bytes):
            buf = io.BytesIO(audio_field)
            arr, sr = sf.read(buf, dtype="float32")
        else:
            raise ValueError(f"Unsupported audio field type: {type(audio_field)}")

        # Convert stereo → mono
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample if needed
        if sr != self.target_sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        return arr.astype(np.float32), sr

    # ── Private: Metadata ─────────────────────────────────────────────────────

    def _build_metadata(self, row: dict) -> dict:
        province_code = str(row.get("province", "")).lower().replace(" ", "")
        return {
            "filename": row.get("filename", ""),
            "transcript": row.get("text", ""),
            "speaker": row.get("speakerID", ""),
            "gender": int(row.get("gender", -1)),
            "province": row.get("province", ""),
            "region": row.get("region", PROVINCE_TO_REGION.get(province_code, "Unknown")),
            "ethnicity": "viet",          # ViMD is Vietnamese; ethnic override in ethnic loader
            "language": "vi",
            "duration": float(row.get("length", 0.0)),
            "set": row.get("set", ""),
        }

    # ── Statistics ────────────────────────────────────────────────────────────

    def print_statistics(self, dataset: DatasetDict) -> None:
        """Print dataset statistics per split."""
        for split, ds in dataset.items():
            df = ds.to_pandas()
            print(f"\n{'='*50}")
            print(f" Split: {split} ({len(df):,} samples)")
            print(f"{'='*50}")
            print(f"  Total duration : {df['length'].sum() / 3600:.2f} hours")
            print(f"  Provinces      : {df['province'].nunique()}")
            print(f"  Speakers       : {df['speakerID'].nunique()}")
            print(f"  Avg duration   : {df['length'].mean():.2f}s")
            print(f"  Region counts  :")
            for region, count in df["region"].value_counts().items():
                print(f"    {region:<10}: {count:,}")


class ViMDTorchDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping a HuggingFace ViMD split.
    Returns dict with 'input_values', 'labels', and metadata.
    """

    def __init__(self, hf_dataset: Dataset, target_sr: int = 16_000):
        self.dataset = hf_dataset
        self.target_sr = target_sr
        self._loader = ViMDLoader(target_sr=target_sr)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        audio_array, sr = self._loader._decode_audio(row["audio"])
        meta = self._loader._build_metadata(row)
        return {
            "input_values": torch.tensor(audio_array, dtype=torch.float32),
            "sampling_rate": sr,
            "transcript": meta["transcript"],
            "province": meta["province"],
            "region": meta["region"],
            "speaker": meta["speaker"],
            "gender": meta["gender"],
            "duration": meta["duration"],
        }


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ViMD Dataset Loader CLI")
    parser.add_argument(
        "--data-dir",
        default="/mnt/d/FPTU/Semester 7/DAT301m/Project/ViMD/data",
        help="Path to ViMD parquet directory",
    )
    parser.add_argument("--use-hf", action="store_true", help="Load from HuggingFace Hub")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loader = ViMDLoader(
        data_dir=args.data_dir,
        use_hf=args.use_hf,
        max_samples=args.max_samples,
    )
    dataset = loader.load_all()
    if args.stats:
        loader.print_statistics(dataset)
    else:
        split = dataset[args.split]
        print(f"Loaded {len(split):,} samples from split '{args.split}'")
        print(f"Columns: {split.column_names}")


if __name__ == "__main__":
    main()
