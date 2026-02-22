"""Decoder modules for Attention U-Net architecture.

This module provides the decoder components for the Attention U-Net generator:
- DecoderBlock: Upsampling block with attention-gated skip connections
- MultiHeadOutput: Multiple 1x1 conv heads for marker-specific outputs

The decoder progressively upsamples features while incorporating attention-weighted
skip connections from the encoder, following the Attention U-Net architecture.
"""

from typing import Optional

import torch
import torch.nn as nn

from multiplex.models.attention import AttentionGate


class DecoderBlock(nn.Module):
    """Single decoder block with optional attention-gated skip connection.

    This block:
    1. Upsamples the input features by 2x
    2. Applies attention gating to the skip connection (if use_attention=True)
    3. Concatenates upsampled features with attention-weighted skip
    4. Applies two 3x3 convolutions with BatchNorm and ReLU

    When use_attention=False, skip connections are concatenated directly
    without attention gating (standard U-Net behavior).

    Now includes optional Dropout2d for MC Dropout uncertainty estimation.

    Attributes:
        upsample: Bilinear upsampling layer (scale_factor=2).
        attention: AttentionGate for filtering skip features (None if use_attention=False).
        conv: Sequential convolutions after concatenation.
        dropout_p: Dropout probability (0.0 means no dropout).
        use_attention: Whether attention gates are enabled.

    Example:
        >>> block = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=256)
        >>> x = torch.randn(1, 1024, 16, 16)   # From previous decoder level
        >>> skip = torch.randn(1, 512, 32, 32)  # From encoder
        >>> out = block(x, skip)
        >>> out.shape
        torch.Size([1, 256, 32, 32])

        >>> # Without attention (standard U-Net)
        >>> block_no_attn = DecoderBlock(1024, 512, 256, use_attention=False)
        >>> out = block_no_attn(x, skip)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        use_attention: bool = True,
    ):
        """Initialize DecoderBlock.

        Args:
            in_channels: Number of channels from previous decoder level (gating signal).
            skip_channels: Number of channels from encoder skip connection.
            out_channels: Number of output channels after this block.
            dropout_p: Dropout probability. Default 0.0 (no dropout).
                Used for MC Dropout uncertainty estimation. Recommended 0.2-0.3.
            use_attention: Whether to use attention gates on skip connections.
                Default True. Set False for standard U-Net skip concatenation.
        """
        super().__init__()

        self.dropout_p = dropout_p
        self.use_attention = use_attention

        # Upsample by 2x using bilinear interpolation
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Attention gate: only create if use_attention=True
        if use_attention:
            self.attention = AttentionGate(
                g_channels=in_channels,
                s_channels=skip_channels,
                inter_channels=max(skip_channels // 2, 1),  # Ensure at least 1 channel
            )
        else:
            self.attention = None  # No attention gate

        # Convolutions after concatenation with optional dropout
        # Input channels = in_channels (upsampled) + skip_channels (attention-weighted)
        layers = [
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Process decoder input with optional attention-gated skip connection.

        Args:
            x: Features from previous decoder level, shape (B, in_channels, H, W).
            skip: Skip features from encoder, shape (B, skip_channels, H', W').
                Expected to be 2x spatial resolution of x.

        Returns:
            Processed features, shape (B, out_channels, H', W').
            Output has same spatial resolution as skip connection.
        """
        # 1. Upsample x to match skip spatial size
        x_up = self.upsample(x)

        # Handle potential size mismatch due to odd input dimensions
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = nn.functional.interpolate(
                x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        # 2. Apply attention to skip connection (if enabled)
        if self.use_attention and self.attention is not None:
            skip_attended = self.attention(x_up, skip)
        else:
            skip_attended = skip  # Direct skip connection (no attention)

        # 3. Concatenate upsampled features with attention-weighted skip
        x_cat = torch.cat([x_up, skip_attended], dim=1)

        # 4. Apply convolutions
        return self.conv(x_cat)

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention coefficients from last forward pass.

        Returns:
            Attention map of shape (B, 1, H, W) with values in [0, 1],
            or None if forward() hasn't been called yet or attention is disabled.
        """
        if self.attention is not None:
            return self.attention.get_attention_map()
        return None  # No attention map when attention disabled


class MultiHeadOutput(nn.Module):
    """Multiple output heads for marker-specific predictions.

    Implements 5 separate 1x1 convolution heads, one for each marker.
    This allows independent gradient flow and marker-specific optimization
    while sharing all features up to the final layer.

    Per CONTEXT.md: "Shared decoder until final layer, 5 separate 1x1
    convolutions for marker-specific output heads."

    Attributes:
        heads: ModuleList of 1x1 Conv2d layers, one per marker.

    Example:
        >>> output = MultiHeadOutput(in_channels=64, num_markers=5)
        >>> x = torch.randn(1, 64, 512, 512)
        >>> out = output(x)
        >>> out.shape
        torch.Size([1, 5, 512, 512])
    """

    def __init__(self, in_channels: int, num_markers: int = 5):
        """Initialize MultiHeadOutput.

        Args:
            in_channels: Number of input channels from final decoder layer.
            num_markers: Number of marker outputs (default 5 for our markers).
        """
        super().__init__()

        self.num_markers = num_markers

        # Create separate 1x1 conv head for each marker
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels, 1, kernel_size=1)
            for _ in range(num_markers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all output heads and concatenate results.

        Args:
            x: Decoder features, shape (B, in_channels, H, W).

        Returns:
            Multi-marker prediction, shape (B, num_markers, H, W).
        """
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=1)
