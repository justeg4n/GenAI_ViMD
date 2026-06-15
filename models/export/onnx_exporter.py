"""
ONNX Model Exporter
====================
Exports trained PyTorch models to:
  - ONNX (with optional quantization)
  - TorchScript
  - GGUF (for llama.cpp / mobile)

Supports: Whisper ASR, Dialect Classifier, mBART Translator
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export trained models to ONNX format for optimized inference.

    Usage:
        exporter = ONNXExporter(output_dir="storage/exports")
        exporter.export_whisper("storage/weights/asr/", opset=17)
        exporter.export_dialect_classifier("storage/weights/dialect/")
    """

    def __init__(
        self,
        output_dir: str = "storage/exports",
        opset_version: int = 17,
        optimize: bool = True,
        quantize: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opset_version = opset_version
        self.optimize = optimize
        self.quantize = quantize

    # ── Whisper Export ────────────────────────────────────────────────────────

    def export_whisper(
        self,
        model_path: str,
        model_name: str = "whisper-large-v3",
        device: str = "cpu",
    ) -> dict[str, Path]:
        """Export Whisper to ONNX using HuggingFace Optimum."""
        output_paths = {}
        try:
            from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
            from transformers import AutoProcessor

            out_dir = self.output_dir / "whisper_onnx"
            processor = AutoProcessor.from_pretrained(model_path or "openai/whisper-large-v3")
            ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_path or "openai/whisper-large-v3",
                export=True,
            )
            ort_model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            logger.info(f"Whisper ONNX saved to {out_dir}")

            output_paths["onnx_dir"] = out_dir
            if self.quantize:
                q_path = self._quantize_onnx_dir(out_dir)
                output_paths["quantized_dir"] = q_path
        except Exception as e:
            logger.error(f"Whisper ONNX export failed: {e}")
        return output_paths

    # ── Dialect Classifier Export ─────────────────────────────────────────────

    def export_dialect_classifier(
        self,
        model_path: str,
        sample_rate: int = 16_000,
        max_duration_s: float = 10.0,
    ) -> Optional[Path]:
        """Export WavLM dialect classifier to ONNX."""
        try:
            import onnx
            from models.dialect.dialect_classifier import DialectClassifierInference

            clf = DialectClassifierInference.from_pretrained(model_path, device="cpu")
            model = clf.model.cpu().eval()

            # Dummy input (batch=1, 10 seconds at 16kHz)
            n_samples = int(max_duration_s * sample_rate)
            dummy_input = torch.randn(1, n_samples)

            output_path = self.output_dir / "dialect_classifier.onnx"
            torch.onnx.export(
                model,
                (dummy_input,),
                str(output_path),
                input_names=["input_values"],
                output_names=["province_logits", "region_logits", "ethnic_logits"],
                dynamic_axes={
                    "input_values": {0: "batch_size", 1: "sequence_length"},
                    "province_logits": {0: "batch_size"},
                    "region_logits": {0: "batch_size"},
                    "ethnic_logits": {0: "batch_size"},
                },
                opset_version=self.opset_version,
                do_constant_folding=True,
                export_params=True,
            )

            if self.optimize:
                self._optimize_onnx(output_path)
            if self.quantize:
                self._quantize_onnx(output_path)

            logger.info(f"Dialect classifier ONNX saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Dialect classifier ONNX export failed: {e}")
            return None

    # ── TorchScript Export ────────────────────────────────────────────────────

    def export_torchscript(
        self,
        model: torch.nn.Module,
        model_name: str,
        dummy_input: torch.Tensor,
    ) -> Optional[Path]:
        """Export a PyTorch model to TorchScript format."""
        try:
            output_path = self.output_dir / f"{model_name}.pt"
            model = model.cpu().eval()
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(output_path))
            logger.info(f"TorchScript saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return None

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _optimize_onnx(self, model_path: Path) -> None:
        """Optimize ONNX model using onnxoptimizer."""
        try:
            import onnx
            import onnxoptimizer
            model = onnx.load(str(model_path))
            passes = ["eliminate_unused_initializer", "fuse_bn_into_conv",
                      "fuse_add_bias_into_conv", "fuse_consecutive_squeezes"]
            optimized = onnxoptimizer.optimize(model, passes)
            onnx.save(optimized, str(model_path))
            logger.info(f"ONNX model optimized: {model_path}")
        except ImportError:
            logger.debug("onnxoptimizer not available, skipping optimization")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")

    def _quantize_onnx(self, model_path: Path) -> Path:
        """Quantize ONNX model to INT8 for faster inference."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q_path = model_path.with_suffix("_int8.onnx")
            quantize_dynamic(str(model_path), str(q_path), weight_type=QuantType.QInt8)
            logger.info(f"Quantized ONNX saved: {q_path}")
            return q_path
        except Exception as e:
            logger.warning(f"ONNX quantization failed: {e}")
            return model_path

    def _quantize_onnx_dir(self, model_dir: Path) -> Path:
        """Quantize all ONNX files in a directory."""
        q_dir = model_dir.parent / f"{model_dir.name}_int8"
        q_dir.mkdir(exist_ok=True)
        for onnx_file in model_dir.glob("*.onnx"):
            self._quantize_onnx(onnx_file)
        return q_dir

    def validate_onnx(self, model_path: Path) -> bool:
        """Validate an ONNX model."""
        try:
            import onnx
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            logger.info(f"✓ ONNX model valid: {model_path}")
            return True
        except Exception as e:
            logger.error(f"✗ ONNX model invalid: {model_path}: {e}")
            return False
