"""Multi-stain input encoder with variable input handling.

This module provides components for handling variable input stain configurations
(IHC-only vs IHC+Hematoxylin) with FiLM-based availability conditioning.

The key innovation is the ability to train on full inputs while still handling
missing modalities at inference time through modality dropout during training
and availability embeddings that condition the network on which inputs are present.

Classes:
    MultiStainInputProjection: Projects variable input channels to fixed dimension
        with FiLM conditioning based on availability.
    ModalityDropout: Randomly drops input modalities during training to teach
        robustness to missing inputs at inference.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MultiStainInputProjection(nn.Module):
    """Project variable input stains to fixed encoder dimension with FiLM conditioning.

    This module handles inputs with different numbers of channels (3 for IHC-only,
    4 for IHC+Hematoxylin) and produces a consistent output representation. The
    availability of different input modalities is communicated through FiLM
    (Feature-wise Linear Modulation) conditioning.

    FiLM applies the transformation: y = gamma * x + beta, where gamma and beta
    are learned based on the input configuration index.

    Input configurations:
        - 0: IHC-only (3 channels RGB)
        - 1: IHC+H (4 channels: RGB + Hematoxylin grayscale)
        - 2: Full (same as 1, used for explicit full-config mode)

    Attributes:
        channel_proj: Conv2d layer projecting max_input_channels to embed_dim.
        avail_embed: Embedding layer producing FiLM gamma/beta parameters.
        norm: BatchNorm2d for normalizing projected features.

    Example:
        >>> proj = MultiStainInputProjection(max_input_channels=4, embed_dim=64)
        >>> x_ihc = torch.randn(2, 3, 512, 512)  # IHC-only
        >>> x_full = torch.randn(2, 4, 512, 512)  # IHC+H
        >>> config_ihc = torch.tensor([0, 0])
        >>> config_full = torch.tensor([1, 1])
        >>> out_ihc = proj(x_ihc, config_ihc)
        >>> out_full = proj(x_full, config_full)
        >>> out_ihc.shape, out_full.shape
        (torch.Size([2, 64, 512, 512]), torch.Size([2, 64, 512, 512]))
    """

    def __init__(
        self,
        max_input_channels: int = 4,
        embed_dim: int = 64,
        num_configs: int = 3,
    ):
        """Initialize MultiStainInputProjection.

        Args:
            max_input_channels: Maximum number of input channels. Default 4
                for IHC RGB (3) + Hematoxylin grayscale (1).
            embed_dim: Output embedding dimension. Default 64, which is the
                typical stem dimension for U-Net architectures.
            num_configs: Number of input configuration modes. Default 3
                for [IHC-only, IHC+H, full].
        """
        super().__init__()

        self.max_input_channels = max_input_channels
        self.embed_dim = embed_dim
        self.num_configs = num_configs

        # Channel projection: conv with large kernel for initial spatial context
        self.channel_proj = nn.Conv2d(
            max_input_channels,
            embed_dim,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        # Availability embedding: produces gamma and beta for FiLM conditioning
        # Output size is embed_dim * 2 to chunk into gamma and beta
        self.avail_embed = nn.Embedding(num_configs, embed_dim * 2)

        # Normalization before FiLM conditioning
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        avail_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project inputs with FiLM availability conditioning.

        Args:
            x: Input tensor of shape (B, C, H, W) where C is 3 or 4.
                Missing channels (when C < max_input_channels) will be zero-padded.
            avail_mask: Configuration indices of shape (B,) with values in
                [0=IHC-only, 1=IHC+H, 2=full].

        Returns:
            Projected features of shape (B, embed_dim, H, W) with FiLM conditioning
            applied based on the availability configuration.
        """
        B, C, H, W = x.shape

        # Pad to max channels if input has fewer channels
        if C < self.max_input_channels:
            padding = torch.zeros(
                B,
                self.max_input_channels - C,
                H,
                W,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, padding], dim=1)

        # Project to embedding dimension and normalize
        h = self.channel_proj(x)
        h = self.norm(h)

        # Get FiLM parameters from availability embedding
        film_params = self.avail_embed(avail_mask)  # (B, embed_dim * 2)
        gamma, beta = film_params.chunk(2, dim=1)  # (B, embed_dim) each

        # Reshape for broadcasting: (B, embed_dim) -> (B, embed_dim, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM: y = gamma * x + beta
        return gamma * h + beta


class ModalityDropout(nn.Module):
    """Randomly drop input modalities during training for robustness.

    During training, this module randomly drops the Hematoxylin channel to
    teach the model to handle missing inputs at inference time. At evaluation,
    no dropout is applied and the full input is passed through.

    This implements "modality dropout" - a standard technique for training
    models that need to work with variable input configurations.

    Input configurations produced:
        - 0: IHC-only (Hematoxylin channel zeroed)
        - 1: IHC+H (all channels present)
        - 2: Full (same as 1, used in eval mode)

    Attributes:
        p_ihc_only: Probability of IHC-only configuration (H channel dropped).
        p_ihc_h: Probability of IHC+H configuration (all channels kept).
        CONFIG_CHANNELS: Mapping from config index to valid channel indices.

    Example:
        >>> dropout = ModalityDropout(p_ihc_only=0.3, p_ihc_h=0.4)
        >>> x = torch.randn(8, 4, 64, 64)  # Full input
        >>> dropout.train()
        >>> masked, configs = dropout(x)
        >>> configs  # Random mix of 0, 1, 2
        tensor([1, 0, 2, 1, 0, 1, 2, 0])
        >>> dropout.eval()
        >>> masked_eval, configs_eval = dropout(x)
        >>> configs_eval  # All 2 (full config)
        tensor([2, 2, 2, 2, 2, 2, 2, 2])
    """

    # Configuration index to valid channel indices
    # Config 0: IHC-only (channels 0, 1, 2 = RGB)
    # Config 1: IHC+H (channels 0, 1, 2, 3 = RGB + Hematoxylin)
    # Config 2: Full (same as 1)
    CONFIG_CHANNELS = {
        0: [0, 1, 2],  # RGB IHC only
        1: [0, 1, 2, 3],  # RGB IHC + Hematoxylin
        2: [0, 1, 2, 3],  # Full (same as 1)
    }

    def __init__(
        self,
        p_ihc_only: float = 0.3,
        p_ihc_h: float = 0.4,
    ):
        """Initialize ModalityDropout.

        Args:
            p_ihc_only: Probability of IHC-only configuration (Hematoxylin
                channel will be zeroed). Default 0.3.
            p_ihc_h: Probability of IHC+H configuration (all channels kept).
                Default 0.4.

        Note:
            The remaining probability (1 - p_ihc_only - p_ihc_h) is assigned
            to the "full" configuration, which is equivalent to IHC+H but
            uses config index 2 for explicit full-config mode.
        """
        super().__init__()

        if p_ihc_only + p_ihc_h > 1.0:
            raise ValueError(
                f"p_ihc_only ({p_ihc_only}) + p_ihc_h ({p_ihc_h}) must be <= 1.0"
            )

        self.p_ihc_only = p_ihc_only
        self.p_ihc_h = p_ihc_h

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply modality dropout during training.

        Args:
            x: Input tensor of shape (B, 4, H, W) with all channels present.

        Returns:
            Tuple of:
                - masked_x: Input with potentially zeroed Hematoxylin channel,
                    shape (B, 4, H, W).
                - config_indices: Configuration index per batch element,
                    shape (B,) with values in {0, 1, 2}.

        Note:
            During eval mode, returns unchanged input with config_indices = 2.
        """
        B = x.shape[0]
        device = x.device

        if not self.training:
            # Evaluation mode: no dropout, return full config
            config_indices = torch.full(
                (B,), 2, device=device, dtype=torch.long
            )
            return x, config_indices

        # Training mode: sample configuration per batch element
        rand = torch.rand(B, device=device)

        # Determine config based on probability thresholds
        # rand < p_ihc_only -> config 0
        # p_ihc_only <= rand < p_ihc_only + p_ihc_h -> config 1
        # rand >= p_ihc_only + p_ihc_h -> config 2
        config_indices = torch.zeros(B, device=device, dtype=torch.long)
        config_indices = torch.where(
            rand < self.p_ihc_only,
            torch.tensor(0, device=device, dtype=torch.long),
            torch.where(
                rand < self.p_ihc_only + self.p_ihc_h,
                torch.tensor(1, device=device, dtype=torch.long),
                torch.tensor(2, device=device, dtype=torch.long),
            ),
        )

        # Create mask: 1 for channels to keep, 0 for channels to drop
        # Only config 0 (IHC-only) drops the Hematoxylin channel (index 3)
        mask = torch.ones_like(x)

        # Zero out channel 3 (Hematoxylin) for samples with config 0
        ihc_only_mask = config_indices == 0
        if ihc_only_mask.any():
            # Expand boolean mask to match spatial dimensions
            # ihc_only_mask: (B,) -> need to mask channel 3 for True elements
            mask[ihc_only_mask, 3] = 0

        return x * mask, config_indices
