"""Attention Gate module for skip connection filtering.

This module implements the additive attention gate mechanism from the
Attention U-Net architecture. The gate learns to focus on relevant
spatial regions by combining decoder context (gating signal) with
encoder features (skip connection).

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention Gate for filtering skip connections.

    Computes additive attention between a gating signal from the decoder
    and skip features from the encoder. The attention coefficients
    highlight relevant spatial regions in the skip features.

    Architecture:
        1. Transform gating signal: W_g(g) -> inter_channels
        2. Transform skip connection: W_s(s) -> inter_channels
        3. Additive attention: ReLU(W_g(g) + W_s(s))
        4. Attention coefficients: psi(attention) -> sigmoid
        5. Apply to skip: s * attention

    Attributes:
        W_g: Transform gating signal to intermediate channels.
        W_s: Transform skip features to intermediate channels.
        psi: Compute attention coefficients from combined features.
        relu: ReLU activation for additive attention.
        last_attention: Stores attention coefficients from last forward pass
            for visualization (shape: B, 1, H, W).

    Example:
        >>> attn = AttentionGate(g_channels=1024, s_channels=512, inter_channels=256)
        >>> g = torch.randn(1, 1024, 16, 16)  # Decoder features
        >>> s = torch.randn(1, 512, 32, 32)   # Encoder skip
        >>> out = attn(g, s)
        >>> out.shape
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, g_channels: int, s_channels: int, inter_channels: int):
        """Initialize AttentionGate.

        Args:
            g_channels: Number of channels in gating signal (from decoder).
            s_channels: Number of channels in skip connection (from encoder).
            inter_channels: Number of intermediate channels for attention
                computation. Typically s_channels // 2.
        """
        super().__init__()

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Transform skip connection
        self.W_s = nn.Sequential(
            nn.Conv2d(s_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

        # Store last attention for visualization
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, g: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted skip features.

        Args:
            g: Gating signal from decoder, shape (B, g_channels, H_g, W_g).
                May have different spatial size than skip - will be interpolated.
            s: Skip features from encoder, shape (B, s_channels, H_s, W_s).

        Returns:
            Attention-weighted skip features, shape (B, s_channels, H_s, W_s).
            Same shape as input skip connection.
        """
        # Upsample gating signal if needed to match skip spatial size
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(
                g, size=s.shape[2:], mode="bilinear", align_corners=False
            )

        # Additive attention
        g_transformed = self.W_g(g)
        s_transformed = self.W_s(s)
        attention = self.relu(g_transformed + s_transformed)
        attention = self.psi(attention)

        # Store for visualization (detached to avoid memory growth)
        self.last_attention = attention.detach()

        # Apply attention to skip features
        return s * attention

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention coefficients from last forward pass.

        Returns:
            Attention map of shape (B, 1, H, W) with values in [0, 1],
            or None if forward() hasn't been called yet.
        """
        return self.last_attention
