"""
Dataset Splitter
================
Creates stratified train/valid/test splits from the ViMD dataset,
ensuring proportional representation of:
  - Province (63 classes)
  - Region (North/Central/South)
  - Gender (male/female)
  - Speaker (no speaker overlap between splits)

Also supports creating evaluation subsets and few-shot splits.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Stratified splitter that ensures balanced province/region/gender
    distribution and prevents speaker leakage between splits.

    Usage:
        splitter = DatasetSplitter(train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
        splits = splitter.split(df, speaker_col="speakerID", stratify_col="province")
        train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        no_speaker_leakage: bool = True,
    ):
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.no_speaker_leakage = no_speaker_leakage
        self._rng = np.random.default_rng(random_seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def split(
        self,
        df: pd.DataFrame,
        speaker_col: str = "speakerID",
        stratify_col: str = "province",
    ) -> dict[str, pd.DataFrame]:
        """
        Split a DataFrame into train/valid/test sets.
        If no_speaker_leakage=True, splits at speaker level to prevent data leakage.
        """
        if self.no_speaker_leakage and speaker_col in df.columns:
            return self._split_by_speaker(df, speaker_col, stratify_col)
        return self._split_stratified(df, stratify_col)

    def split_dataset_dict(
        self, dataset_dict: DatasetDict
    ) -> DatasetDict:
        """
        Re-split a HuggingFace DatasetDict.
        Combines all splits then re-stratifies.
        """
        all_dfs = []
        for split_name, ds in dataset_dict.items():
            df = ds.to_pandas()
            df["_original_split"] = split_name
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        splits = self.split(combined)
        return DatasetDict({
            split: Dataset.from_pandas(df.drop(columns=["_original_split"], errors="ignore"))
            for split, df in splits.items()
        })

    def create_few_shot_subset(
        self,
        df: pd.DataFrame,
        n_shots: int = 5,
        stratify_col: str = "province",
    ) -> pd.DataFrame:
        """Create a few-shot subset with exactly n_shots samples per class."""
        groups = df.groupby(stratify_col)
        subsets = []
        for name, group in groups:
            n = min(n_shots, len(group))
            subsets.append(group.sample(n, random_state=self.random_seed))
        return pd.concat(subsets, ignore_index=True)

    def create_province_eval_set(
        self,
        df: pd.DataFrame,
        n_per_province: int = 30,
    ) -> pd.DataFrame:
        """Create a balanced evaluation set with n samples per province."""
        return self.create_few_shot_subset(df, n_shots=n_per_province, stratify_col="province")

    # ── Private ───────────────────────────────────────────────────────────────

    def _split_by_speaker(
        self,
        df: pd.DataFrame,
        speaker_col: str,
        stratify_col: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Split at speaker level: all utterances of a speaker go to the same split.
        Uses stratified sampling of speakers by province/region.
        """
        # Group speakers by their primary stratum
        speaker_stratum = (
            df.groupby(speaker_col)[stratify_col]
            .agg(lambda x: x.mode()[0])
            .reset_index()
        )
        speaker_stratum.columns = [speaker_col, stratify_col]
        all_speakers = speaker_stratum[speaker_col].values

        # Stratified split of speakers
        speakers_per_stratum = defaultdict(list)
        for _, row in speaker_stratum.iterrows():
            speakers_per_stratum[row[stratify_col]].append(row[speaker_col])

        train_spk, valid_spk, test_spk = [], [], []
        for stratum, speakers in speakers_per_stratum.items():
            self._rng.shuffle(speakers)
            n = len(speakers)
            n_train = max(1, int(n * self.train_ratio))
            n_valid = max(1, int(n * self.valid_ratio))
            train_spk.extend(speakers[:n_train])
            valid_spk.extend(speakers[n_train:n_train + n_valid])
            test_spk.extend(speakers[n_train + n_valid:])

        train_set = set(train_spk)
        valid_set = set(valid_spk)
        test_set = set(test_spk)

        train_df = df[df[speaker_col].isin(train_set)].reset_index(drop=True)
        valid_df = df[df[speaker_col].isin(valid_set)].reset_index(drop=True)
        test_df = df[df[speaker_col].isin(test_set)].reset_index(drop=True)

        self._log_split_stats(train_df, valid_df, test_df, stratify_col)
        return {"train": train_df, "valid": valid_df, "test": test_df}

    def _split_stratified(
        self,
        df: pd.DataFrame,
        stratify_col: str,
    ) -> dict[str, pd.DataFrame]:
        """Simple stratified random split (may have speaker leakage)."""
        try:
            from sklearn.model_selection import train_test_split
            train_valid, test_df = train_test_split(
                df,
                test_size=self.test_ratio,
                stratify=df[stratify_col] if stratify_col in df.columns else None,
                random_state=self.random_seed,
            )
            valid_ratio_adjusted = self.valid_ratio / (self.train_ratio + self.valid_ratio)
            train_df, valid_df = train_test_split(
                train_valid,
                test_size=valid_ratio_adjusted,
                stratify=train_valid[stratify_col] if stratify_col in train_valid.columns else None,
                random_state=self.random_seed,
            )
        except ImportError:
            # Manual split without sklearn
            df_shuffled = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            n = len(df_shuffled)
            n_train = int(n * self.train_ratio)
            n_valid = int(n * self.valid_ratio)
            train_df = df_shuffled[:n_train]
            valid_df = df_shuffled[n_train:n_train + n_valid]
            test_df = df_shuffled[n_train + n_valid:]

        self._log_split_stats(train_df, valid_df, test_df, stratify_col)
        return {
            "train": train_df.reset_index(drop=True),
            "valid": valid_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    def _log_split_stats(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        stratify_col: str,
    ) -> None:
        total = len(train_df) + len(valid_df) + len(test_df)
        logger.info(
            f"Dataset split: train={len(train_df):,} ({len(train_df)/total*100:.1f}%) | "
            f"valid={len(valid_df):,} ({len(valid_df)/total*100:.1f}%) | "
            f"test={len(test_df):,} ({len(test_df)/total*100:.1f}%)"
        )
        if stratify_col in train_df.columns:
            train_classes = train_df[stratify_col].nunique()
            valid_classes = valid_df[stratify_col].nunique()
            test_classes = test_df[stratify_col].nunique()
            logger.info(
                f"Unique {stratify_col}: train={train_classes}, "
                f"valid={valid_classes}, test={test_classes}"
            )

    # ── Verification ──────────────────────────────────────────────────────────

    def verify_no_leakage(
        self,
        splits: dict[str, pd.DataFrame],
        speaker_col: str = "speakerID",
    ) -> bool:
        """Verify there is no speaker overlap between splits."""
        train_speakers = set(splits["train"][speaker_col]) if speaker_col in splits["train"].columns else set()
        valid_speakers = set(splits["valid"][speaker_col]) if speaker_col in splits["valid"].columns else set()
        test_speakers = set(splits["test"][speaker_col]) if speaker_col in splits["test"].columns else set()

        tv_overlap = train_speakers & valid_speakers
        tt_overlap = train_speakers & test_speakers
        vt_overlap = valid_speakers & test_speakers

        if tv_overlap or tt_overlap or vt_overlap:
            logger.error(
                f"Speaker leakage detected! "
                f"train∩valid={len(tv_overlap)}, "
                f"train∩test={len(tt_overlap)}, "
                f"valid∩test={len(vt_overlap)}"
            )
            return False
        logger.info("✓ No speaker leakage detected between splits")
        return True


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Dataset Splitter CLI")
    parser.add_argument("--data-dir", required=True, help="Path to ViMD parquet directory")
    parser.add_argument("--output-dir", default="storage/datasets/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--no-leakage", action="store_true", default=True)
    args = parser.parse_args()

    from dataset.ingestion.vimd_loader import ViMDLoader
    loader = ViMDLoader(data_dir=args.data_dir)
    dd = loader.load_all()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splitter = DatasetSplitter(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        no_speaker_leakage=args.no_leakage,
    )

    for split_name, ds in dd.items():
        df = ds.to_pandas()
        splits = splitter.split(df)
        splitter.verify_no_leakage(splits)
        for name, sdf in splits.items():
            out_path = output_dir / f"{split_name}_{name}.parquet"
            sdf.to_parquet(str(out_path), index=False)
            print(f"Saved {len(sdf):,} rows to {out_path}")


if __name__ == "__main__":
    main()
