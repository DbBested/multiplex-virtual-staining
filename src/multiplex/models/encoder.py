"""ConvNeXt encoder wrapper with multi-scale feature extraction.

This module provides a wrapper around timm's ConvNeXt-Base model for
extracting hierarchical features at multiple scales, suitable for
encoder-decoder architectures like Attention U-Net.

The encoder extracts features at 4 stages with channels [128, 256, 512, 1024]
and spatial reductions [4, 8, 16, 32] relative to input resolution.
"""

from typing import List, Tuple

import timm
import torch
import torch.nn as nn


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-Base encoder with multi-scale feature extraction.

    Wraps timm's ConvNeXt-Base model with `features_only=True` to extract
    hierarchical features at 4 scales. Supports grayscale input via the
    `in_channels` parameter - timm automatically adapts pretrained weights.

    Attributes:
        encoder: The underlying timm ConvNeXt model.
        _channels: Tuple of channel counts at each stage.
        _reductions: Tuple of spatial reductions at each stage.

    Example:
        >>> encoder = ConvNeXtEncoder(in_channels=1, pretrained=True)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> features = encoder(x)
        >>> [f.shape for f in features]
        [torch.Size([1, 128, 128, 128]),
         torch.Size([1, 256, 64, 64]),
         torch.Size([1, 512, 32, 32]),
         torch.Size([1, 1024, 16, 16])]
    """

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        model_name: str = "convnext_base.fb_in22k_ft_in1k",
    ):
        """Initialize ConvNeXt encoder.

        Args:
            in_channels: Number of input channels. Default 1 for grayscale BF.
                timm automatically adapts pretrained weights for non-RGB input.
            pretrained: Whether to load ImageNet pretrained weights.
            model_name: timm model name. Default uses ImageNet-22k pretraining
                fine-tuned on ImageNet-1k for best transfer learning.
        """
        super().__init__()

        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3),  # All 4 stages
        )

        # Cache feature info for decoder construction
        self._channels = tuple(self.encoder.feature_info.channels())
        self._reductions = tuple(self.encoder.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from input.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            List of 4 feature tensors at progressively coarser scales:
            - Stage 0: (B, 128, H/4, W/4)
            - Stage 1: (B, 256, H/8, W/8)
            - Stage 2: (B, 512, H/16, W/16)
            - Stage 3: (B, 1024, H/32, W/32)
        """
        return self.encoder(x)

    def get_channels(self) -> Tuple[int, ...]:
        """Get channel counts at each encoder stage.

        Returns:
            Tuple of channel counts: (128, 256, 512, 1024) for ConvNeXt-Base.
        """
        return self._channels

    def get_reductions(self) -> Tuple[int, ...]:
        """Get spatial reductions (strides) at each encoder stage.

        Returns:
            Tuple of reductions: (4, 8, 16, 32) for ConvNeXt-Base.
        """
        return self._reductions

    @property
    def out_channels(self) -> Tuple[int, ...]:
        """Alias for get_channels() for compatibility."""
        return self._channels
