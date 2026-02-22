"""Perceptual losses for flow matching training (PixelGen recipe).

Provides P-DINO loss using frozen DINOv2 patch features for global semantic
alignment. Complements LPIPS which captures local textures/sharpness.

Reference:
    PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss
    (zehong-ma.github.io/PixelGen)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PDINOLoss(nn.Module):
    """P-DINO perceptual loss using frozen DINOv2 patch features.

    Compares patch-level DINOv2 representations between predicted and target
    images using cosine distance. Captures global semantic structure (spatial
    layout, where nuclei should be) while LPIPS captures local textures.

    The backbone is frozen and never updated. Gradients flow through the
    predicted image path only, back to the generator.

    Args:
        model_name: timm model name for DINOv2 variant.
            Default: ViT-S/14 (22M params, memory-efficient).
        device: Device to place the backbone on.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ImageNet normalization (DINOv2 expects this)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Hook onto final norm to capture all patch tokens
        self._patch_tokens = None
        self.backbone.norm.register_forward_hook(self._capture_hook)

        num_params = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"P-DINO loss initialized: {model_name} ({num_params / 1e6:.1f}M params)"
        )

    def _capture_hook(self, module, input, output):
        """Forward hook to capture patch tokens from final norm layer."""
        # output shape: (B, N+1, D) where N = num_patches, +1 for CLS token
        self._patch_tokens = output[:, 1:, :]  # Remove CLS, keep patches

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute P-DINO loss between predicted and target images.

        Args:
            pred: Predicted images (B, 3, H, W) in [0, 1]. Gradients flow
                through this path back to the generator.
            target: Target images (B, 3, H, W) in [0, 1]. Detached.

        Returns:
            Scalar cosine distance loss averaged over patches and batch.
        """
        # Normalize to ImageNet stats and resize to 518x518 (DINOv2 patch_size=14)
        pred_n = (pred.float().clamp(0, 1) - self.mean) / self.std
        target_n = (target.float().clamp(0, 1) - self.mean) / self.std
        pred_n = F.interpolate(pred_n, size=518, mode="bilinear", align_corners=False)
        target_n = F.interpolate(target_n, size=518, mode="bilinear", align_corners=False)

        # Forward pred (with grad for generator backprop)
        self.backbone(pred_n)
        pred_patches = self._patch_tokens  # (B, N, D)

        # Forward target (no grad -- just reference features)
        with torch.no_grad():
            self.backbone(target_n)
            target_patches = self._patch_tokens  # (B, N, D)

        # Cosine distance on L2-normalized patch features
        pred_patches = F.normalize(pred_patches, dim=-1)
        target_patches = F.normalize(target_patches, dim=-1)
        loss = (1 - F.cosine_similarity(pred_patches, target_patches, dim=-1)).mean()
        return loss
