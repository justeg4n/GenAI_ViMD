"""
Dialect Classifier — WavLM-Large + Attention Pooling
=====================================================
Province-level dialect identification across 63 Vietnamese provinces.

Architecture:
  WavLM-Large encoder → Attention Pooling → Multi-head classification
    - 63-class province head (+ ArcFace)
    - 3-class region head (North/Central/South)
    - 3-class ethnic head (viet/ede/bahnar)
    - Code-switching score

Outputs:
  {province, region, confidence, ethnic_probability, code_switching_score}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, WavLMModel

logger = logging.getLogger(__name__)

# ── Province / Region Constants ────────────────────────────────────────────────
PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu",
    "Bắc Ninh", "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước",
    "Bình Thuận", "Cà Mau", "Cần Thơ", "Cao Bằng", "Đà Nẵng",
    "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp",
    "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội", "Hà Tĩnh",
    "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên",
    "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng",
    "Lạng Sơn", "Lào Cai", "Long An", "Nam Định", "Nghệ An",
    "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình",
    "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng",
    "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa",
    "Thừa Thiên Huế", "Tiền Giang", "TP. Hồ Chí Minh", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
]
assert len(PROVINCES) == 63, f"Expected 63 provinces, got {len(PROVINCES)}"

REGIONS = ["North", "Central", "South"]
ETHNICITIES = ["viet", "ede", "bahnar"]

PROVINCE_TO_REGION = {
    # North
    **{p: "North" for p in [
        "Hà Nội", "Hà Nam", "Hưng Yên", "Ninh Bình", "Nam Định",
        "Thái Bình", "Bắc Ninh", "Bắc Giang", "Hải Dương", "Hải Phòng",
        "Quảng Ninh", "Lạng Sơn", "Cao Bằng", "Bắc Kạn", "Thái Nguyên",
        "Tuyên Quang", "Hà Giang", "Lào Cai", "Yên Bái", "Phú Thọ",
        "Vĩnh Phúc", "Hòa Bình", "Sơn La", "Điện Biên", "Lai Châu",
    ]},
    # Central
    **{p: "Central" for p in [
        "Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị",
        "Thừa Thiên Huế", "Đà Nẵng", "Quảng Nam", "Quảng Ngãi", "Bình Định",
        "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận",
        "Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng",
    ]},
    # South
    **{p: "South" for p in [
        "TP. Hồ Chí Minh", "Bình Dương", "Đồng Nai", "Bà Rịa - Vũng Tàu",
        "Long An", "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long",
        "Đồng Tháp", "An Giang", "Kiên Giang", "Cần Thơ", "Hậu Giang",
        "Sóc Trăng", "Bạc Liêu", "Cà Mau", "Bình Phước", "Tây Ninh",
    ]},
}


# ── Attention Pooling ─────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Attention-weighted temporal pooling.
    Maps (B, T, D) → (B, D) using learned attention weights.
    """

    def __init__(self, input_dim: int, attention_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            mask: (B, T) boolean, True = valid, False = padding
        Returns:
            pooled: (B, D)
        """
        attn_weights = self.attention(x).squeeze(-1)      # (B, T)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)    # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled


# ── ArcFace Loss ──────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for improved class separability.
    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss"
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.3,
        scale: float = 64.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.threshold = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: normalized embeddings (B, D)
            labels: ground truth class indices (B,)
        Returns:
            loss scalar
        """
        x = F.normalize(x, p=2, dim=-1)
        w = F.normalize(self.weight, p=2, dim=-1)
        cosine = F.linear(x, w)                           # (B, C)

        # ArcFace: add angular margin
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m     # cos(θ + m)
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(-1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance across 63 provinces."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ── Dialect Classifier Model ──────────────────────────────────────────────────

class DialectClassifier(nn.Module):
    """
    Multi-task dialect classifier based on WavLM-Large.

    Outputs per inference:
      - province logits: (B, 63)
      - region logits:   (B, 3)
      - ethnic logits:   (B, 3)
      - embeddings:      (B, D) — for speaker/dialect distance
    """

    def __init__(
        self,
        encoder_model: str = "microsoft/wavlm-large",
        num_province_classes: int = 63,
        num_region_classes: int = 3,
        num_ethnic_classes: int = 3,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        pooling: str = "attention",
        use_arcface: bool = True,
        arcface_margin: float = 0.3,
        arcface_scale: float = 64.0,
        freeze_encoder_layers: int = 12,
    ):
        super().__init__()
        self.num_province_classes = num_province_classes
        self.num_region_classes = num_region_classes
        self.num_ethnic_classes = num_ethnic_classes
        self.use_arcface = use_arcface

        # ── Encoder ──
        config = AutoConfig.from_pretrained(encoder_model)
        self.encoder = WavLMModel.from_pretrained(encoder_model)
        encoder_dim = config.hidden_size

        # Freeze bottom layers for stability
        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # ── Pooling ──
        if pooling == "attention":
            self.pooling = AttentionPooling(encoder_dim)
        else:
            self.pooling = None  # Mean pooling fallback

        # ── Projection ──
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Classification Heads ──
        self.province_head = nn.Linear(hidden_size, num_province_classes)
        self.region_head = nn.Linear(hidden_size, num_region_classes)
        self.ethnic_head = nn.Linear(hidden_size, num_ethnic_classes)

        # ── ArcFace ──
        if use_arcface:
            self.arcface = ArcFaceLoss(
                in_features=hidden_size,
                num_classes=num_province_classes,
                margin=arcface_margin,
                scale=arcface_scale,
            )

        # ── Loss ──
        self.province_loss_fn = FocalLoss(gamma=2.0)
        self.region_loss_fn = nn.CrossEntropyLoss()
        self.ethnic_loss_fn = nn.CrossEntropyLoss()

    def encode(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run encoder + pooling → embedding (B, hidden_size)."""
        output = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = output.last_hidden_state  # (B, T, D)

        if self.pooling is not None:
            mask = None
            if attention_mask is not None:
                # Map input mask to hidden state length
                mask = attention_mask.bool()
                if mask.shape[1] != hidden.shape[1]:
                    mask = mask[:, :hidden.shape[1]]
            pooled = self.pooling(hidden, mask)
        else:
            pooled = hidden.mean(dim=1)

        return self.proj(pooled)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        province_labels: Optional[torch.Tensor] = None,
        region_labels: Optional[torch.Tensor] = None,
        ethnic_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        embeddings = self.encode(input_values, attention_mask)

        province_logits = self.province_head(embeddings)
        region_logits = self.region_head(embeddings)
        ethnic_logits = self.ethnic_head(embeddings)

        output = {
            "province_logits": province_logits,
            "region_logits": region_logits,
            "ethnic_logits": ethnic_logits,
            "embeddings": embeddings,
        }

        if province_labels is not None:
            loss_province = self.province_loss_fn(province_logits, province_labels)
            loss_region = self.region_loss_fn(region_logits, region_labels) \
                          if region_labels is not None else torch.tensor(0.0)
            loss_ethnic = self.ethnic_loss_fn(ethnic_logits, ethnic_labels) \
                          if ethnic_labels is not None else torch.tensor(0.0)

            arcface_loss = torch.tensor(0.0)
            if self.use_arcface and province_labels is not None:
                arcface_loss = self.arcface(embeddings, province_labels)

            total_loss = (
                1.0 * loss_province +
                0.3 * loss_region +
                0.2 * loss_ethnic +
                0.5 * arcface_loss
            )
            output["loss"] = total_loss
            output["loss_province"] = loss_province
            output["loss_region"] = loss_region
            output["loss_ethnic"] = loss_ethnic
            output["loss_arcface"] = arcface_loss

        return output


# ── Inference Wrapper ─────────────────────────────────────────────────────────

class DialectClassifierInference:
    """
    Production inference wrapper for dialect classification.

    Usage:
        clf = DialectClassifierInference.from_pretrained("storage/weights/dialect/")
        result = clf.predict(audio_array, sample_rate=16000)
        # result = {"province": "Hà Nội", "region": "North", "confidence": 0.92,
        #           "ethnic_probability": {"viet": 0.98, "ede": 0.01, "bahnar": 0.01}}
    """

    def __init__(
        self,
        model: DialectClassifier,
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = model.to(self.device).eval()
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "auto",
        **kwargs,
    ) -> "DialectClassifierInference":
        """Load from a saved checkpoint directory."""
        path = Path(model_path)
        # Load config
        import yaml
        config_path = path / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        model = DialectClassifier(**config.get("model", {}))
        weights_path = path / "model.pt"
        if weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded dialect model weights from {weights_path}")
        else:
            logger.warning(f"No weights found at {weights_path} — using random weights")
        return cls(model=model, device=device, **kwargs)

    @torch.inference_mode()
    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
        top_k: int = 3,
    ) -> dict:
        """
        Run dialect identification on an audio array.

        Returns:
            {
              "province": str,
              "region": str,
              "confidence": float,
              "top_k_provinces": [(province, prob), ...],
              "ethnic_probability": {"viet": float, "ede": float, "bahnar": float},
              "code_switching_score": float,
            }
        """
        # Preprocess
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward
        output = self.model(input_values=audio_tensor)

        province_probs = F.softmax(output["province_logits"], dim=-1).squeeze(0).cpu().numpy()
        region_probs = F.softmax(output["region_logits"], dim=-1).squeeze(0).cpu().numpy()
        ethnic_probs = F.softmax(output["ethnic_logits"], dim=-1).squeeze(0).cpu().numpy()

        # Top province
        top_idx = int(np.argmax(province_probs))
        confidence = float(province_probs[top_idx])
        province = PROVINCES[top_idx] if top_idx < len(PROVINCES) else f"Province_{top_idx}"
        region = PROVINCE_TO_REGION.get(province, REGIONS[int(np.argmax(region_probs))])

        # Top-k provinces
        top_k_indices = np.argsort(province_probs)[::-1][:top_k]
        top_k_provinces = [
            (PROVINCES[i] if i < len(PROVINCES) else f"Province_{i}", float(province_probs[i]))
            for i in top_k_indices
        ]

        # Ethnic probabilities
        ethnic_probability = {
            eth: float(ethnic_probs[i]) for i, eth in enumerate(ETHNICITIES)
        }

        # Code-switching score: entropy-based
        ethnic_entropy = float(-np.sum(ethnic_probs * np.log(ethnic_probs + 1e-8)))
        code_switching_score = float(ethnic_entropy / np.log(len(ETHNICITIES)))

        return {
            "province": province,
            "region": region,
            "confidence": confidence,
            "top_k_provinces": top_k_provinces,
            "ethnic_probability": ethnic_probability,
            "code_switching_score": code_switching_score,
        }

    def predict_batch(
        self,
        audio_list: list[np.ndarray],
        sample_rate: int = 16_000,
    ) -> list[dict]:
        """Batch prediction for efficiency."""
        return [self.predict(audio, sample_rate) for audio in audio_list]
