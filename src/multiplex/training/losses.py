"""
Multi-loss module for pix2pix-style GAN training.

This module provides loss functions for generator and discriminator training:
- MultiLoss: Combined L1 + perceptual (LPIPS) + adversarial loss for generator
- DiscriminatorLoss: Real/fake classification loss for discriminator
- V3PatchNCELoss: PatchNCE wrapper for V3Generator feature extraction

The loss weights follow pix2pix conventions:
- lambda_l1 = 100.0 (reconstruction loss)
- lambda_perc = 10.0 (perceptual/feature matching)
- lambda_gan = 1.0 (adversarial)

PatchNCE (contrastive) loss can optionally be added for misalignment-robust training.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import lpips

from multiplex.training.patchnce import PatchNCELoss
from multiplex.training.biological_constraints import (
    BiologicalConstraintLoss,
    BiologicalConstraintConfig,
)


class MultiLoss(nn.Module):
    """Combined loss for generator training.

    Computes L1 reconstruction + LPIPS perceptual + adversarial losses.
    Optionally includes PatchNCE contrastive loss for misalignment-robust training.

    The perceptual loss uses a pretrained VGG network via the lpips library.
    Since our markers are single-channel, we average across markers and
    expand to 3 channels for LPIPS input.

    Args:
        lambda_l1: Weight for L1 reconstruction loss. Default 100.0.
        lambda_perc: Weight for perceptual loss. Default 10.0.
        lambda_gan: Weight for adversarial loss. Default 1.0.
        lambda_bio: Weight for biological constraint loss. Default 0.0 (disabled).
        bio_config: Configuration for biological constraints. Default None.
        device: Device for LPIPS network. Default 'cuda'.
        use_patchnce: Enable PatchNCE contrastive loss. Default False.
        nce_layers: Encoder layer indices for PatchNCE. Default (0, 2).
        nce_t: Temperature for InfoNCE softmax. Default 0.07.
        num_patches: Number of patches to sample per layer. Default 256.
        lambda_nce: Weight for NCE loss. Default 1.0.

    Example:
        >>> criterion = MultiLoss(lambda_l1=100.0, lambda_perc=10.0)
        >>> fake = torch.randn(4, 5, 256, 256)
        >>> real = torch.randn(4, 5, 256, 256)
        >>> pred_fake = torch.randn(4, 1, 30, 30)
        >>> total, loss_dict = criterion(fake, real, pred_fake)

    Example with PatchNCE:
        >>> criterion = MultiLoss(use_patchnce=True, nce_layers=(0, 2))
        >>> criterion.init_patchnce((128, 256, 512, 1024))  # encoder channels
        >>> nce_loss, nce_dict = criterion.forward_nce(feat_q, feat_k)

    Example with biological constraints:
        >>> from multiplex.training.biological_constraints import BiologicalConstraintConfig
        >>> bio_cfg = BiologicalConstraintConfig()
        >>> criterion = MultiLoss(lambda_bio=0.1, bio_config=bio_cfg)
        >>> total, loss_dict = criterion(fake, real, pred_fake)
        >>> # loss_dict includes: loss_bio, loss_excl, loss_contain, loss_coloc
    """

    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_perc: float = 10.0,
        lambda_gan: float = 1.0,
        lambda_bio: float = 0.0,  # Biological constraint weight (0 = disabled)
        bio_config: Optional[BiologicalConstraintConfig] = None,
        device: str = "cuda",
        # PatchNCE options
        use_patchnce: bool = False,
        nce_layers: Tuple[int, ...] = (0, 2),
        nce_t: float = 0.07,
        num_patches: int = 256,
        lambda_nce: float = 1.0,
        # Per-marker LPIPS option
        per_marker_lpips: bool = False,  # Compute LPIPS per marker instead of averaged
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_gan = lambda_gan
        self.lambda_bio = lambda_bio
        self.per_marker_lpips = per_marker_lpips

        # L1 reconstruction loss
        self.criterion_l1 = nn.L1Loss()

        # Adversarial loss (BCE with logits)
        self.criterion_gan = nn.BCEWithLogitsLoss()

        # Perceptual loss with VGG features
        # Use spatial=False for global perceptual loss (returns single value)
        self.criterion_perc = lpips.LPIPS(net="vgg", spatial=False)
        self.criterion_perc.eval()
        # Freeze LPIPS weights - no gradient computation needed
        for param in self.criterion_perc.parameters():
            param.requires_grad = False

        # PatchNCE loss (optional, for misalignment-robust training)
        self.use_patchnce = use_patchnce
        self.patchnce_loss = None
        if use_patchnce:
            self.patchnce_loss = PatchNCELoss(
                nce_layers=list(nce_layers),
                nce_t=nce_t,
                num_patches=num_patches,
                lambda_nce=lambda_nce,
            )

        # Biological constraint loss (optional, for biological plausibility)
        self.bio_loss = None
        if lambda_bio > 0:
            self.bio_loss = BiologicalConstraintLoss(
                bio_config or BiologicalConstraintConfig()
            )

        # Move to device
        self._device = device

    def to(self, device):
        """Override to ensure LPIPS network and bio loss move with module."""
        self._device = device
        self.criterion_perc = self.criterion_perc.to(device)
        if self.bio_loss is not None:
            self.bio_loss = self.bio_loss.to(device)
        return super().to(device)

    def _prepare_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input for LPIPS (expects 3-channel in [-1, 1]).

        Our markers are (B, 5, H, W). We average across markers to get
        (B, 1, H, W), then expand to (B, 3, H, W) for VGG input.

        Args:
            x: Input tensor of shape (B, C, H, W) where C is number of markers.

        Returns:
            Tensor of shape (B, 3, H, W) suitable for LPIPS.
        """
        # Average across marker channels: (B, 5, H, W) -> (B, 1, H, W)
        x_avg = x.mean(dim=1, keepdim=True)
        # Expand to 3 channels: (B, 1, H, W) -> (B, 3, H, W)
        x_rgb = x_avg.expand(-1, 3, -1, -1)
        return x_rgb

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor, pred_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined generator loss.

        Args:
            fake: Generated markers, shape (B, num_markers, H, W).
            real: Ground truth markers, shape (B, num_markers, H, W).
            pred_fake: Discriminator output for fake images, shape (B, 1, H', W').

        Returns:
            Tuple of:
            - total_loss: Weighted sum of all losses (scalar tensor).
            - loss_dict: Dictionary with individual loss values (floats).
        """
        # L1 reconstruction loss
        loss_l1 = self.criterion_l1(fake, real)

        # Perceptual loss (LPIPS)
        if self.per_marker_lpips:
            # Compute LPIPS per marker for better per-channel perceptual quality
            num_markers = fake.shape[1]
            marker_lpips = []
            for i in range(num_markers):
                # Extract single marker and expand to 3 channels
                fake_marker = fake[:, i:i+1, :, :].expand(-1, 3, -1, -1)
                real_marker = real[:, i:i+1, :, :].expand(-1, 3, -1, -1)
                marker_loss = self.criterion_perc(fake_marker, real_marker).mean()
                marker_lpips.append(marker_loss)
            loss_perc = torch.stack(marker_lpips).mean()
        else:
            # Original: average markers then compute LPIPS
            fake_rgb = self._prepare_for_lpips(fake)
            real_rgb = self._prepare_for_lpips(real)
            # LPIPS returns per-sample distance, take mean across batch
            loss_perc = self.criterion_perc(fake_rgb, real_rgb).mean()

        # Adversarial loss - generator wants discriminator to output 1 (real)
        # Create target tensor of ones matching discriminator output shape
        target_real = torch.ones_like(pred_fake)
        loss_gan = self.criterion_gan(pred_fake, target_real)

        # Combined loss
        total_loss = (
            self.lambda_gan * loss_gan
            + self.lambda_l1 * loss_l1
            + self.lambda_perc * loss_perc
        )

        loss_dict = {
            "loss_l1": loss_l1.item(),
            "loss_perc": loss_perc.item(),
            "loss_gan": loss_gan.item(),
        }

        # Add biological constraints if enabled
        if self.bio_loss is not None and self.lambda_bio > 0:
            bio_loss, bio_dict = self.bio_loss(fake, real)
            total_loss = total_loss + self.lambda_bio * bio_loss
            loss_dict.update(bio_dict)

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict

    def init_patchnce(self, encoder_channels: Tuple[int, ...]) -> None:
        """Initialize PatchNCE MLP heads with encoder channel info.

        Must be called before forward_nce() when use_patchnce=True.

        Args:
            encoder_channels: Tuple of channel counts at each encoder stage.
                For example, (128, 256, 512, 1024) for ConvNeXt-Base.
        """
        if self.patchnce_loss is not None:
            self.patchnce_loss.init_mlp_heads(encoder_channels)

    def forward_nce(
        self,
        feat_q: List[torch.Tensor],
        feat_k: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PatchNCE loss between encoder features.

        This should be called separately from forward() in the training loop,
        as it requires encoder features rather than final outputs.

        Args:
            feat_q: Query features from input BF encoding (list of encoder stage outputs).
            feat_k: Key features from predicted marker encoding (list of encoder stage outputs).

        Returns:
            Tuple of:
            - NCE loss tensor (scalar) or zero tensor if PatchNCE disabled.
            - Loss dict with 'loss_nce' key.
        """
        if self.patchnce_loss is None:
            return torch.tensor(0.0, device=feat_q[0].device), {"loss_nce": 0.0}

        loss_nce = self.patchnce_loss(feat_q, feat_k)
        return loss_nce, {"loss_nce": loss_nce.item()}


class DiscriminatorLoss(nn.Module):
    """Loss for discriminator training.

    Computes binary cross-entropy loss for real/fake classification.
    Optionally applies label smoothing for training stability.

    Args:
        label_smoothing: Amount of label smoothing. Default 0.0.
            When > 0, real labels become (1 - smoothing) and
            fake labels become smoothing.

    Example:
        >>> criterion = DiscriminatorLoss(label_smoothing=0.1)
        >>> pred_real = torch.randn(4, 1, 30, 30)
        >>> pred_fake = torch.randn(4, 1, 30, 30)
        >>> loss, loss_dict = criterion(pred_real, pred_fake)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, pred_real: torch.Tensor, pred_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute discriminator loss.

        Args:
            pred_real: Discriminator output for real images, shape (B, 1, H', W').
            pred_fake: Discriminator output for fake images, shape (B, 1, H', W').

        Returns:
            Tuple of:
            - d_loss: Total discriminator loss (scalar tensor).
            - loss_dict: Dictionary with individual loss values (floats).
        """
        # Target labels with optional smoothing
        if self.label_smoothing > 0:
            real_label = 1.0 - self.label_smoothing
            fake_label = self.label_smoothing
        else:
            real_label = 1.0
            fake_label = 0.0

        # Create target tensors
        target_real = torch.full_like(pred_real, real_label)
        target_fake = torch.full_like(pred_fake, fake_label)

        # Compute losses
        loss_d_real = self.criterion(pred_real, target_real)
        loss_d_fake = self.criterion(pred_fake, target_fake)

        # Average of real and fake losses
        d_loss = 0.5 * (loss_d_real + loss_d_fake)

        loss_dict = {
            "loss_d_real": loss_d_real.item(),
            "loss_d_fake": loss_d_fake.item(),
            "loss_d": d_loss.item(),
        }

        return d_loss, loss_dict


class V3PatchNCELoss(nn.Module):
    """PatchNCE loss wrapper for V3Generator feature extraction.

    Computes contrastive loss between input encoding and output encoding
    features, enabling misalignment-robust training.

    The loss encourages structural consistency between input and output
    even when pixel alignment is imperfect. This is particularly useful
    for the DeepLIIF dataset where pseudo-pairs may have registration noise.

    This wrapper simplifies PatchNCE usage with V3Generator by:
    1. Automatically initializing MLP heads from generator encoder channels
    2. Providing a simple forward() interface with feature lists
    3. Handling device placement and initialization checks

    Args:
        nce_layers: Encoder layer indices to use. Default [0, 2] uses
            stage 0 (1/4 resolution) and stage 2 (1/16 resolution).
        nce_t: Temperature for InfoNCE softmax. Default 0.07.
        num_patches: Number of patches to sample per layer. Default 256.
        lambda_nce: Loss weight. Default 1.0.

    Example:
        >>> loss_fn = V3PatchNCELoss()
        >>> loss_fn.init_for_generator(generator)  # Initialize from generator
        >>> output, feat_q = generator.forward_with_features(input)
        >>> # Re-encode output through generator for key features
        >>> _, feat_k = generator.forward_with_features(output_for_encode)
        >>> nce_loss = loss_fn(feat_q, feat_k)

    Note:
        For typical usage with V3Generator, you would:
        1. Create the loss: `loss_fn = V3PatchNCELoss()`
        2. Initialize from generator: `loss_fn.init_for_generator(generator)`
        3. In training loop:
           - `out, feat_q = generator.forward_with_features(x)`
           - `_, feat_k = generator.forward_with_features(x)` (or different input)
           - `nce_loss = loss_fn(feat_q, feat_k)`
    """

    def __init__(
        self,
        nce_layers: Optional[List[int]] = None,
        nce_t: float = 0.07,
        num_patches: int = 256,
        lambda_nce: float = 1.0,
    ):
        """Initialize V3PatchNCELoss.

        Args:
            nce_layers: Encoder layer indices to use. Default [0, 2].
            nce_t: Temperature for InfoNCE. Default 0.07.
            num_patches: Patches to sample per layer. Default 256.
            lambda_nce: Loss weight. Default 1.0.
        """
        super().__init__()

        if nce_layers is None:
            nce_layers = [0, 2]

        self.nce_layers = nce_layers
        self.nce_t = nce_t
        self.num_patches = num_patches
        self.lambda_nce = lambda_nce

        # Create underlying PatchNCE loss
        self.patchnce = PatchNCELoss(
            nce_layers=nce_layers,
            nce_t=nce_t,
            num_patches=num_patches,
            lambda_nce=lambda_nce,
        )

        self._initialized = False

    def init_for_generator(self, generator: nn.Module) -> None:
        """Initialize MLP heads based on generator's encoder channels.

        This must be called before forward() to set up the projection
        heads for each selected encoder layer.

        Args:
            generator: V3Generator or any model with get_encoder_channels() method.

        Raises:
            AttributeError: If generator lacks get_encoder_channels() method.

        Example:
            >>> from multiplex.models import V3Generator
            >>> generator = V3Generator(pretrained=False)
            >>> loss_fn = V3PatchNCELoss()
            >>> loss_fn.init_for_generator(generator)
        """
        channels = generator.get_encoder_channels()
        self.patchnce.init_mlp_heads(channels)
        self._initialized = True

    def forward(
        self,
        feat_q: List[torch.Tensor],
        feat_k: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute PatchNCE loss between query and key features.

        Args:
            feat_q: Query features from input encoding. List of encoder
                stage outputs from forward_with_features().
            feat_k: Key features from output encoding. List of encoder
                stage outputs from forward_with_features().

        Returns:
            Scalar tensor with weighted PatchNCE loss.

        Raises:
            RuntimeError: If init_for_generator() was not called.

        Example:
            >>> output, feat_q = generator.forward_with_features(x)
            >>> _, feat_k = generator.forward_with_features(x)
            >>> loss = loss_fn(feat_q, feat_k)
            >>> loss.backward()
        """
        if not self._initialized:
            raise RuntimeError(
                "Call init_for_generator() before forward(). "
                "Example: loss_fn.init_for_generator(generator)"
            )

        return self.patchnce(feat_q, feat_k)

    def to(self, device):
        """Override to ensure PatchNCE module moves with wrapper."""
        self.patchnce = self.patchnce.to(device)
        return super().to(device)
