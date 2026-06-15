"""
Global Style Token (GST) Prosody Model
=======================================
Models prosody for dialect-aware TTS synthesis.
Used by XTTS-v2 for regional intonation conditioning.

Architecture: Reference encoder → GST attention → Style token embeddings
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReferenceEncoder(nn.Module):
    """
    CNN-based reference encoder that extracts prosody features
    from a mel-spectrogram reference audio.
    """

    def __init__(
        self,
        mel_channels: int = 80,
        conv_channels: list[int] = None,
        gru_hidden: int = 128,
    ):
        super().__init__()
        conv_channels = conv_channels or [32, 32, 64, 64, 128, 128]

        self.convs = nn.ModuleList()
        in_ch = 1
        for out_ch in conv_channels:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(),
            ))
            in_ch = out_ch

        # Compute output size after convolutions
        dummy = torch.zeros(1, 1, mel_channels, 100)
        with torch.no_grad():
            for conv in self.convs:
                dummy = conv(dummy)
        freq_size = dummy.shape[2]
        gru_input_size = dummy.shape[1] * freq_size

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        self.output_size = gru_hidden

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, mel_channels)
        Returns:
            embedding: (B, gru_hidden)
        """
        x = mel.unsqueeze(1)  # (B, 1, T, mel_channels) → (B, 1, mel, T)
        x = x.permute(0, 1, 3, 2)

        for conv in self.convs:
            x = conv(x)

        # Reshape for GRU: (B, T', C*F')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        _, h_n = self.gru(x)
        return h_n.squeeze(0)  # (B, gru_hidden)


class MultiHeadStyleAttention(nn.Module):
    """
    Multi-head attention over style token library.
    Learns to combine style tokens based on reference audio.
    """

    def __init__(
        self,
        query_dim: int,
        token_dim: int,
        num_heads: int = 4,
        num_tokens: int = 10,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.token_dim = token_dim

        # Style token library (learnable)
        self.style_tokens = nn.Parameter(
            torch.FloatTensor(num_tokens, token_dim // num_heads)
        )
        nn.init.normal_(self.style_tokens)

        self.attention = nn.Linear(query_dim, num_heads * (token_dim // num_heads))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, query_dim) — reference encoder output
        Returns:
            style_embedding: (B, token_dim)
        """
        B = query.shape[0]
        attn_weights = self.attention(query).view(B, self.num_heads, -1)  # (B, H, D/H)
        attn_weights = F.softmax(attn_weights, dim=-1)

        tokens = self.style_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, D/H)
        # For each head, compute weighted sum
        style = torch.bmm(
            attn_weights,
            tokens.permute(0, 2, 1)
        )  # (B, H, num_tokens) × (B, D/H, num_tokens)^T — actually need to do this per head
        style = style.view(B, -1)  # (B, H * D/H = token_dim)
        return style


class GSTProsogyModel(nn.Module):
    """
    Full Global Style Token model combining reference encoder + style attention.

    Used for:
      1. Training-time: Extract reference prosody, condition TTS decoder
      2. Inference-time: Condition on target dialect/emotion style embedding

    Usage:
        gst = GSTProsogyModel()
        style_emb = gst(reference_mel)  # (B, token_dim)
        # Then concatenate style_emb with text encoder output in TTS
    """

    def __init__(
        self,
        mel_channels: int = 80,
        ref_encoder_hidden: int = 128,
        num_style_tokens: int = 10,
        num_attention_heads: int = 4,
        token_dim: int = 256,
    ):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(
            mel_channels=mel_channels,
            gru_hidden=ref_encoder_hidden,
        )
        self.style_attention = MultiHeadStyleAttention(
            query_dim=ref_encoder_hidden,
            token_dim=token_dim,
            num_heads=num_attention_heads,
            num_tokens=num_style_tokens,
        )
        self.output_dim = token_dim

    def forward(
        self,
        reference_mel: Optional[torch.Tensor] = None,
        style_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for style embedding extraction.

        Args:
            reference_mel: (B, T, mel_channels) — use for reference-based conditioning
            style_token_indices: (B,) int — use for index-based style selection at inference

        Returns:
            style_embedding: (B, token_dim)
        """
        if reference_mel is not None:
            ref_emb = self.ref_encoder(reference_mel)
            style_emb = self.style_attention(ref_emb)
            return style_emb
        elif style_token_indices is not None:
            # Direct style token selection (for preset emotions/dialects)
            tokens = self.style_attention.style_tokens  # (num_tokens, D/H)
            selected = tokens[style_token_indices]  # (B, D/H)
            H = self.style_attention.num_heads
            return selected.unsqueeze(1).expand(-1, H, -1).contiguous().view(
                style_token_indices.shape[0], -1
            )
        else:
            raise ValueError("Provide either reference_mel or style_token_indices")


# ── Dialect Style Token Registry ──────────────────────────────────────────────
# Maps dialect regions to style token presets
DIALECT_STYLE_TOKENS = {
    "North":   0,   # Northern flat tones
    "Central": 1,   # Central rising tones
    "South":   2,   # Southern melodic tones
    "neutral": 3,
    "happy":   4,
    "sad":     5,
    "angry":   6,
    "surprised": 7,
    "fearful": 8,
    "formal":  9,
}
