"""Source encoder wrapper for conditioning the JiT model.

This module wraps V3's ConvNeXt encoder and MultiStainInputProjection to create
source tokens for cross-attention conditioning in the ConditionalJiT model.

The source encoder:
1. Projects variable input channels (3 for IHC-only, 4 for IHC+H) to 64 dim
2. Extracts multi-scale features using pretrained ConvNeXt-Base
3. Projects stage 3 features (1024 channels, 16x16) to transformer dimension
4. Flattens to 256 tokens for cross-attention

Reference:
    Reuses V3 encoder infrastructure for transfer learning.
"""

import torch
import torch.nn as nn

from multiplex.models.encoder import ConvNeXtEncoder
from multiplex.models.multi_stain_encoder import MultiStainInputProjection


class SourceEncoder(nn.Module):
    """Wrapper around V3's ConvNeXt encoder for source image encoding.

    Takes IHC input (optionally with Hematoxylin channel) and produces
    256 source tokens for cross-attention conditioning in the JiT model.

    The encoder reuses pretrained V3 components:
    - MultiStainInputProjection: Handles variable input channels with FiLM
    - ConvNeXtEncoder: Pretrained ConvNeXt-Base for feature extraction

    Args:
        hidden_size: Dimension of output tokens. Default 768 (JiT-B).
        pretrained: Whether to load pretrained ConvNeXt weights. Default True.
        freeze_encoder: Whether to freeze encoder parameters. Default True.

    Attributes:
        input_proj: MultiStainInputProjection for variable input handling.
        encoder: ConvNeXtEncoder for feature extraction.
        projection: 1x1 conv to project to transformer dimension.

    Example:
        >>> encoder = SourceEncoder(hidden_size=768, pretrained=True)
        >>> x_ihc = torch.randn(2, 3, 512, 512)
        >>> config_ihc = torch.tensor([0, 0])
        >>> tokens = encoder(x_ihc, config_ihc)
        >>> tokens.shape
        torch.Size([2, 256, 768])
    """

    def __init__(
        self,
        hidden_size: int = 768,
        pretrained: bool = True,
        freeze_encoder: bool = True,
    ):
        """Initialize SourceEncoder.

        Args:
            hidden_size: Dimension of output tokens.
            pretrained: Whether to load pretrained ConvNeXt weights.
            freeze_encoder: Whether to freeze encoder parameters.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.freeze_encoder = freeze_encoder

        # Input projection: handles variable input channels (3 or 4)
        # Projects to 64 channels with FiLM conditioning based on availability
        self.input_proj = MultiStainInputProjection(
            max_input_channels=4,
            embed_dim=64,
            num_configs=3,
        )

        # ConvNeXt encoder: pretrained feature extraction
        # Input: (B, 64, 512, 512) -> Features at 4 scales
        # Stage 3: (B, 1024, 16, 16) - 256 spatial positions
        self.encoder = ConvNeXtEncoder(
            in_channels=64,
            pretrained=pretrained,
        )

        # Projection from ConvNeXt stage 3 (1024 channels) to transformer dim
        self.projection = nn.Conv2d(1024, hidden_size, kernel_size=1)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters (input_proj and encoder, not projection)."""
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        avail_config: torch.Tensor,
    ) -> torch.Tensor:
        """Encode source image to tokens for cross-attention.

        Args:
            x: IHC input tensor of shape (B, C, H, W) where C is 3 or 4.
                - 3 channels: IHC RGB only
                - 4 channels: IHC RGB + Hematoxylin grayscale
            avail_config: Availability configuration of shape (B,).
                - 0: IHC-only
                - 1: IHC+H
                - 2: Full (same as 1)

        Returns:
            Source tokens of shape (B, 256, hidden_size).
            256 tokens from 16x16 spatial grid.
        """
        # Project input with FiLM conditioning: (B, C, 512, 512) -> (B, 64, 512, 512)
        h = self.input_proj(x, avail_config)

        # Extract features: (B, 64, 512, 512) -> list of 4 feature maps
        features = self.encoder(h)

        # Use stage 3 (1024 channels, 16x16 for 512x512 input)
        stage3 = features[-1]  # (B, 1024, 16, 16)

        # Project to transformer dimension: (B, 1024, 16, 16) -> (B, hidden_size, 16, 16)
        source = self.projection(stage3)

        # Flatten to tokens: (B, hidden_size, 16, 16) -> (B, 256, hidden_size)
        source = source.flatten(2).transpose(1, 2)

        return source
