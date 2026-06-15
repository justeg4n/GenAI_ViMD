"""
Dataset Exporter
================
Exports processed datasets to multiple formats:
  - HuggingFace DatasetDict (push to hub or save locally)
  - Parquet shards for distributed training
  - Arrow IPC files for fast random access
  - CSV manifest files for lightweight metadata

Supports both ViMD speech data and ethnic language corpora.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Exports processed datasets in multiple formats for training.

    Usage:
        exporter = DatasetExporter(output_dir="storage/datasets/export")
        exporter.export_parquet(dataset_dict)
        exporter.export_hf(dataset_dict, repo_id="myorg/vietspeech")
        exporter.export_manifest(df, "train_manifest.csv")
    """

    def __init__(
        self,
        output_dir: str = "storage/datasets/export",
        shard_size: int = 1000,       # Rows per parquet shard
        compression: str = "snappy",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.compression = compression

    # ── HuggingFace Format ────────────────────────────────────────────────────

    def export_hf_local(
        self,
        dataset_dict: DatasetDict,
        name: str = "vietspeech",
    ) -> Path:
        """Save DatasetDict in HuggingFace format to disk."""
        out_path = self.output_dir / name
        dataset_dict.save_to_disk(str(out_path))
        logger.info(f"Saved HuggingFace DatasetDict to {out_path}")
        return out_path

    def push_to_hub(
        self,
        dataset_dict: DatasetDict,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
    ) -> None:
        """Push dataset to HuggingFace Hub."""
        import os
        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not set — cannot push to hub")
        dataset_dict.push_to_hub(repo_id, token=token, private=private)
        logger.info(f"Pushed dataset to hub: {repo_id}")

    # ── Parquet Export ────────────────────────────────────────────────────────

    def export_parquet(
        self,
        dataset_dict: DatasetDict,
        exclude_audio: bool = False,
    ) -> dict[str, list[Path]]:
        """
        Export all splits as sharded Parquet files.
        Returns dict of {split_name: [shard_paths]}.
        """
        result: dict[str, list[Path]] = {}
        for split_name, dataset in dataset_dict.items():
            df = dataset.to_pandas()
            if exclude_audio and "audio" in df.columns:
                df = df.drop(columns=["audio"])
            shard_paths = self._write_parquet_shards(df, split_name)
            result[split_name] = shard_paths
        return result

    def _write_parquet_shards(
        self, df: pd.DataFrame, split_name: str
    ) -> list[Path]:
        """Write a DataFrame as sharded Parquet files."""
        n_shards = max(1, len(df) // self.shard_size)
        shard_paths = []
        for i, chunk_start in enumerate(range(0, len(df), self.shard_size)):
            chunk = df.iloc[chunk_start:chunk_start + self.shard_size]
            shard_path = self.output_dir / f"{split_name}-{i:05d}-of-{n_shards:05d}.parquet"
            chunk.to_parquet(
                str(shard_path),
                index=False,
                compression=self.compression,
            )
            shard_paths.append(shard_path)
        logger.info(f"Exported {split_name}: {n_shards} shards, {len(df):,} rows")
        return shard_paths

    # ── Arrow Export ──────────────────────────────────────────────────────────

    def export_arrow(
        self,
        df: pd.DataFrame,
        filename: str,
    ) -> Path:
        """Export to Apache Arrow IPC format for fast streaming."""
        out_path = self.output_dir / filename
        table = pa.Table.from_pandas(df)
        with pa.ipc.new_file(str(out_path), table.schema) as writer:
            # Write in chunks for memory efficiency
            for i in range(0, len(table), self.shard_size):
                writer.write_table(table.slice(i, self.shard_size))
        logger.info(f"Saved Arrow file: {out_path} ({len(df):,} rows)")
        return out_path

    # ── CSV Manifest ──────────────────────────────────────────────────────────

    def export_manifest(
        self,
        df: pd.DataFrame,
        filename: str,
        audio_root: Optional[str] = None,
    ) -> Path:
        """
        Export lightweight CSV manifest (no audio bytes).
        Format: filename, transcript, speaker, gender, province, region, duration
        """
        cols = [
            "filename", "text", "speakerID", "gender",
            "province", "region", "length", "set",
        ]
        available = [c for c in cols if c in df.columns]
        manifest = df[available].copy()
        if audio_root:
            manifest["audio_path"] = manifest["filename"].apply(
                lambda f: str(Path(audio_root) / f)
            )
        out_path = self.output_dir / filename
        manifest.to_csv(str(out_path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved manifest: {out_path} ({len(manifest):,} rows)")
        return out_path

    # ── JSON Lines Export ─────────────────────────────────────────────────────

    def export_jsonl(
        self,
        df: pd.DataFrame,
        filename: str,
        include_columns: Optional[list[str]] = None,
    ) -> Path:
        """Export to JSONL format (one JSON object per line)."""
        out_path = self.output_dir / filename
        if include_columns:
            df = df[[c for c in include_columns if c in df.columns]]
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved JSONL: {out_path} ({len(df):,} rows)")
        return out_path

    # ── Statistics Export ─────────────────────────────────────────────────────

    def export_statistics(
        self,
        dataset_dict: DatasetDict,
        filename: str = "dataset_statistics.json",
    ) -> Path:
        """Export dataset statistics as JSON."""
        stats: dict = {}
        for split_name, dataset in dataset_dict.items():
            df = dataset.to_pandas()
            split_stats: dict = {
                "n_samples": len(df),
                "n_speakers": int(df["speakerID"].nunique()) if "speakerID" in df.columns else None,
                "n_provinces": int(df["province"].nunique()) if "province" in df.columns else None,
                "total_duration_hours": float(df["length"].sum() / 3600) if "length" in df.columns else None,
                "avg_duration_s": float(df["length"].mean()) if "length" in df.columns else None,
                "columns": list(df.columns),
            }
            if "province" in df.columns:
                split_stats["province_counts"] = df["province"].value_counts().to_dict()
            if "region" in df.columns:
                split_stats["region_counts"] = df["region"].value_counts().to_dict()
            if "gender" in df.columns:
                split_stats["gender_counts"] = df["gender"].value_counts().to_dict()
            stats[split_name] = split_stats

        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics: {out_path}")
        return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Dataset Exporter CLI")
    parser.add_argument("--data-dir", required=True, help="Path to ViMD parquet directory")
    parser.add_argument("--output-dir", default="storage/datasets/export")
    parser.add_argument("--format", choices=["parquet", "arrow", "manifest", "hf", "all"],
                        default="manifest")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    from dataset.ingestion.vimd_loader import ViMDLoader
    loader = ViMDLoader(data_dir=args.data_dir, max_samples=args.max_samples)
    dd = loader.load_all()

    exporter = DatasetExporter(output_dir=args.output_dir)
    if args.format in ("parquet", "all"):
        exporter.export_parquet(dd, exclude_audio=True)
    if args.format in ("hf", "all"):
        exporter.export_hf_local(dd)
    if args.format in ("manifest", "all"):
        for split_name, ds in dd.items():
            exporter.export_manifest(ds.to_pandas(), f"{split_name}_manifest.csv")
    exporter.export_statistics(dd)
    print(f"Export complete → {args.output_dir}")


if __name__ == "__main__":
    main()
