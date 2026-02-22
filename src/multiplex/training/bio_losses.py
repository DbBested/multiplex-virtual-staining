"""Differentiable biological consistency losses for JiT flow matching training.

This module provides biologically-grounded auxiliary losses that encode known
spatial relationships between virtual staining channels:

1. NuclearConsistencyLoss (BIO-01): Penalizes Lap2 (nuclear envelope) signal
   outside dilated DAPI (nuclei) regions. Uses soft sigmoid thresholding,
   kornia morphological dilation, and optional distance-weighted penalty.

2. SpatialCoherenceLoss (BIO-02): Computes 1 - Pearson(Hematoxylin, DAPI_pred)
   as a loss. Both stain nuclei so they should be spatially correlated.

3. BioLossSuite: Wrapper that combines sub-losses with per-loss weights,
   ablation toggles, and noise gating (only apply when timestep > threshold).

All losses operate on the x0 estimate (predicted clean image) from flow
matching, NOT on the velocity prediction. They require non-detached x1_hat
so gradients flow back through the model.

Channel mapping (JiT pipeline):
    DAPI = channel 0 (nuclear stain)
    Lap2 = channel 1 (nuclear envelope)
    Marker = channel 2 (Ki67 expression)

References:
    - PixelGen (arXiv:2602.02493): Noise gating at t > 0.3
    - Existing biological_constraints.py: Proven patterns for soft thresholding
      and distance-weighted penalty (different channel set, NOT imported)

Example:
    >>> from multiplex.training.bio_losses import BioLossSuite
    >>> import torch
    >>> suite = BioLossSuite()
    >>> x1_hat = torch.rand(4, 3, 64, 64, requires_grad=True)
    >>> target = torch.rand(4, 3, 64, 64)
    >>> hema = torch.rand(4, 1, 64, 64)
    >>> timesteps = torch.tensor([0.5, 0.6, 0.7, 0.8])
    >>> loss, loss_dict = suite(x1_hat, target, hema, timesteps)
    >>> loss.backward()
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from kornia.contrib import distance_transform
    from kornia import morphology as morph

    HAS_KORNIA = True
except ImportError:
    distance_transform = None
    morph = None
    HAS_KORNIA = False

# ---------------------------------------------------------------------------
# Channel constants for JiT 3-channel output
# ---------------------------------------------------------------------------

DAPI_IDX = 0  # Nuclear stain
LAP2_IDX = 1  # Nuclear envelope
MARKER_IDX = 2  # Ki67 expression


# ---------------------------------------------------------------------------
# BIO-01: Nuclear Consistency Loss
# ---------------------------------------------------------------------------


class NuclearConsistencyLoss(nn.Module):
    """Penalize Lap2 signal outside dilated DAPI regions (BIO-01).

    Biological prior: Lap2 (nuclear envelope) is a spatial subset of DAPI
    (nuclei). Any Lap2 signal outside the dilated DAPI mask is a violation.

    L_nuclear = mean(Lap2 * (1 - dilate(sigmoid(k*(DAPI - thresh)))) * weight)

    Uses soft sigmoid thresholding for differentiability and optional
    distance-weighted penalty for smoother gradients.

    Args:
        dilation_kernel_size: Size of square dilation kernel. Default 7.
        threshold: Soft threshold for DAPI binarization. Default 0.3.
        temperature: Sigmoid sharpness (higher = more binary-like). Default 10.0.
        margin_pixels: Distance over which penalty ramps. Default 10.
        use_distance_weighting: Apply distance transform weighting. Default True.

    Example:
        >>> loss_fn = NuclearConsistencyLoss()
        >>> x1_hat = torch.rand(2, 3, 64, 64, requires_grad=True)
        >>> target = torch.rand(2, 3, 64, 64)
        >>> loss, info = loss_fn(x1_hat, target)
        >>> loss.backward()
    """

    def __init__(
        self,
        dilation_kernel_size: int = 7,
        threshold: float = 0.3,
        temperature: float = 10.0,
        margin_pixels: int = 10,
        use_distance_weighting: bool = True,
    ):
        super().__init__()
        if not HAS_KORNIA:
            raise ImportError(
                "kornia required for NuclearConsistencyLoss: "
                "pip install kornia>=0.7.0"
            )

        self.threshold = threshold
        self.temperature = temperature
        self.margin_pixels = margin_pixels
        self.use_distance_weighting = use_distance_weighting

        k = dilation_kernel_size
        self.register_buffer("dilation_kernel", torch.ones(k, k))

    def forward(
        self,
        x1_hat: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute nuclear consistency loss.

        Args:
            x1_hat: Predicted clean image (B, 3, H, W). Ch0=DAPI, Ch1=Lap2.
                    Must be non-detached for gradient flow.
            target: Ground truth (B, 3, H, W). If provided, use GT DAPI
                    for mask (more stable during early training).

        Returns:
            Tuple of (loss, loss_dict) where loss is a scalar tensor and
            loss_dict contains 'loss_nuclear_consistency' key.
        """
        # Cast to float32 for kornia compatibility (may not support bfloat16)
        orig_dtype = x1_hat.dtype
        x1_hat_f32 = x1_hat.float()

        # Use GT DAPI for mask if available (more stable)
        if target is not None:
            dapi = target[:, DAPI_IDX : DAPI_IDX + 1, :, :].float()
        else:
            dapi = x1_hat_f32[:, DAPI_IDX : DAPI_IDX + 1, :, :]

        lap2 = x1_hat_f32[:, LAP2_IDX : LAP2_IDX + 1, :, :]  # (B, 1, H, W)

        # Soft threshold DAPI -> approximate binary mask
        dapi_mask = torch.sigmoid(self.temperature * (dapi - self.threshold))

        # Dilate DAPI mask to create tolerance region
        dapi_dilated = morph.dilation(dapi_mask, self.dilation_kernel)

        # Penalty: Lap2 outside dilated DAPI
        outside_dapi = 1.0 - dapi_dilated

        if self.use_distance_weighting:
            # Distance-weighted penalty: farther from boundary = higher penalty.
            # distance_transform computes distance FROM foreground (non-zero) pixels.
            # We pass the dilated DAPI binary mask as foreground, so pixels OUTSIDE
            # DAPI get high distance values (far from nearest DAPI pixel).
            # Use hard-thresholded mask for distance_transform (soft mask produces
            # incorrect distances). Distance weights are detached -- gradient flows
            # through outside_dapi and lap2 only.
            dapi_dilated_binary = (dapi_dilated.detach() > 0.5).float()
            if dapi_dilated_binary.sum() > 0:
                dist_from_dapi = distance_transform(dapi_dilated_binary)
                penalty_weight = torch.clamp(
                    dist_from_dapi / self.margin_pixels, 0, 1
                )
            else:
                # No DAPI foreground at all -- full penalty everywhere
                penalty_weight = torch.ones_like(outside_dapi)
            violation = lap2 * outside_dapi * penalty_weight
        else:
            violation = lap2 * outside_dapi

        loss = violation.mean()

        return loss, {"loss_nuclear_consistency": loss.item()}


# ---------------------------------------------------------------------------
# BIO-02: Spatial Coherence Loss
# ---------------------------------------------------------------------------


class SpatialCoherenceLoss(nn.Module):
    """H-DAPI Pearson correlation loss (BIO-02).

    Biological prior: Hematoxylin and DAPI both stain nuclei, so they
    should be spatially correlated. Loss = 1 - Pearson(H, DAPI_pred).

    Args:
        eps: Small value to prevent division by zero. Default 1e-8.

    Example:
        >>> loss_fn = SpatialCoherenceLoss()
        >>> x1_hat = torch.rand(2, 3, 64, 64, requires_grad=True)
        >>> hema = torch.rand(2, 1, 64, 64)
        >>> loss, info = loss_fn(x1_hat, hema)
        >>> loss.backward()
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x1_hat: torch.Tensor,
        hematoxylin: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute spatial coherence loss.

        Args:
            x1_hat: Predicted clean image (B, 3, H, W). Ch0=DAPI.
                    Must be non-detached for gradient flow.
            hematoxylin: Source H channel (B, 1, H, W). This is a
                         conditioning input, NOT a target.

        Returns:
            Tuple of (loss, loss_dict) where loss is a scalar tensor and
            loss_dict contains 'loss_spatial_coherence' and 'h_dapi_pearson'.
        """
        dapi_pred = x1_hat[:, DAPI_IDX : DAPI_IDX + 1, :, :]  # (B, 1, H, W)
        h = hematoxylin  # (B, 1, H, W)

        # Flatten spatial dimensions for per-sample Pearson correlation
        B = dapi_pred.shape[0]
        dapi_flat = dapi_pred.reshape(B, -1)  # (B, H*W)
        h_flat = h.reshape(B, -1)  # (B, H*W)

        # Center the signals
        dapi_mean = dapi_flat.mean(dim=1, keepdim=True)
        h_mean = h_flat.mean(dim=1, keepdim=True)

        dapi_c = dapi_flat - dapi_mean
        h_c = h_flat - h_mean

        # Pearson correlation per sample
        numer = (dapi_c * h_c).sum(dim=1)  # (B,)
        # Add eps inside sqrt to prevent inf gradient at sqrt(0).
        # When dapi_pred is all zeros (e.g. zero-init model), sum-of-squares = 0
        # and sqrt'(0) = inf. Even with clamp blocking upstream gradient,
        # PyTorch computes 0 * inf = NaN (IEEE 754), corrupting all weights.
        denom = (
            (dapi_c.pow(2).sum(dim=1) + self.eps).sqrt()
            * (h_c.pow(2).sum(dim=1) + self.eps).sqrt()
        ).clamp(min=self.eps)

        corr = numer / denom  # (B,)

        # Loss = 1 - mean correlation
        loss = 1.0 - corr.mean()

        return loss, {
            "loss_spatial_coherence": loss.item(),
            "h_dapi_pearson": corr.mean().item(),
        }


# ---------------------------------------------------------------------------
# BioLossSuite: Combined wrapper with noise gating and ablation toggles
# ---------------------------------------------------------------------------


class BioLossSuite(nn.Module):
    """Combined biological loss suite for JiT flow matching training.

    Wraps NuclearConsistencyLoss (BIO-01) and SpatialCoherenceLoss (BIO-02)
    with per-loss weights, on/off toggles for ablation studies, and noise
    gating (only apply when timestep > threshold).

    Args:
        weight_nuclear: Weight for nuclear consistency loss. Default 1.0.
        weight_coherence: Weight for spatial coherence loss. Default 1.0.
        enable_nuclear: Enable BIO-01. Default True.
        enable_coherence: Enable BIO-02. Default True.
        t_threshold: Noise gating threshold. Only apply bio losses when
                     timestep > this value. Default 0.3.
        use_gt_dapi: Use GT DAPI for nuclear mask (more stable). Default True.
        dilation_kernel_size: Kernel size for DAPI dilation. Default 7.
        nuclear_threshold: Soft threshold for DAPI binarization. Default 0.3.
        nuclear_temperature: Sigmoid sharpness. Default 10.0.
        use_distance_weighting: Distance-weighted penalty. Default True.
        margin_pixels: Distance margin for penalty ramp. Default 10.

    Example:
        >>> suite = BioLossSuite()
        >>> x1_hat = torch.rand(4, 3, 64, 64, requires_grad=True)
        >>> target = torch.rand(4, 3, 64, 64)
        >>> hema = torch.rand(4, 1, 64, 64)
        >>> timesteps = torch.tensor([0.5, 0.6, 0.7, 0.8])
        >>> loss, loss_dict = suite(x1_hat, target, hema, timesteps)
        >>> loss.backward()
    """

    def __init__(
        self,
        weight_nuclear: float = 1.0,
        weight_coherence: float = 1.0,
        enable_nuclear: bool = True,
        enable_coherence: bool = True,
        t_threshold: float = 0.3,
        use_gt_dapi: bool = True,
        # NuclearConsistencyLoss params
        dilation_kernel_size: int = 7,
        nuclear_threshold: float = 0.3,
        nuclear_temperature: float = 10.0,
        use_distance_weighting: bool = True,
        margin_pixels: int = 10,
    ):
        super().__init__()
        self.weight_nuclear = weight_nuclear
        self.weight_coherence = weight_coherence
        self.enable_nuclear = enable_nuclear
        self.enable_coherence = enable_coherence
        self.t_threshold = t_threshold
        self.use_gt_dapi = use_gt_dapi

        if enable_nuclear:
            self.nuclear_loss = NuclearConsistencyLoss(
                dilation_kernel_size=dilation_kernel_size,
                threshold=nuclear_threshold,
                temperature=nuclear_temperature,
                margin_pixels=margin_pixels,
                use_distance_weighting=use_distance_weighting,
            )

        if enable_coherence:
            self.coherence_loss = SpatialCoherenceLoss()

    def forward(
        self,
        x1_hat: torch.Tensor,
        target: torch.Tensor,
        hematoxylin: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all enabled bio losses with noise gating.

        Args:
            x1_hat: Predicted clean image (B, 3, H, W). Non-detached.
            target: Ground truth (B, 3, H, W).
            hematoxylin: Source H channel (B, 1, H, W). Can be None
                         (BIO-02 skipped in IHC-only mode).
            timesteps: Sampled timesteps (B,) in [0, 1].

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Weighted sum of all bio losses (scalar, requires_grad)
            - loss_dict: Per-loss values for logging with keys:
                loss_bio_total, loss_nuclear_consistency,
                loss_spatial_coherence, bio_gate_fraction, h_dapi_pearson
        """
        device = x1_hat.device
        loss_dict: Dict[str, float] = {}

        # Noise gating: only apply when t > threshold
        # At low t (high noise), x0 estimate is garbage
        mask = (timesteps > self.t_threshold).float()  # (B,)
        gate_fraction = mask.sum() / mask.numel()

        if gate_fraction == 0.0:
            # No samples above threshold -- return zero loss with grad
            zero_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
            loss_dict["loss_bio_total"] = 0.0
            loss_dict["loss_nuclear_consistency"] = 0.0
            loss_dict["loss_spatial_coherence"] = 0.0
            loss_dict["bio_gate_fraction"] = 0.0
            loss_dict["h_dapi_pearson"] = 0.0
            return zero_loss, loss_dict

        # Accumulate weighted losses
        total = torch.tensor(0.0, device=device)

        # BIO-01: Nuclear consistency
        if self.enable_nuclear:
            gt = target if self.use_gt_dapi else None
            nuc_loss, nuc_dict = self.nuclear_loss(x1_hat, gt)
            total = total + self.weight_nuclear * nuc_loss
            loss_dict.update(nuc_dict)
        else:
            loss_dict["loss_nuclear_consistency"] = 0.0

        # BIO-02: Spatial coherence (only when hematoxylin available)
        if self.enable_coherence and hematoxylin is not None:
            coh_loss, coh_dict = self.coherence_loss(x1_hat, hematoxylin)
            total = total + self.weight_coherence * coh_loss
            loss_dict.update(coh_dict)
        else:
            loss_dict["loss_spatial_coherence"] = 0.0
            loss_dict["h_dapi_pearson"] = 0.0

        # Apply noise gate fraction: scale by proportion of valid samples
        gated_total = total * gate_fraction

        loss_dict["loss_bio_total"] = gated_total.item()
        loss_dict["bio_gate_fraction"] = gate_fraction.item()

        return gated_total, loss_dict


# ---------------------------------------------------------------------------
# Convenience exports
# ---------------------------------------------------------------------------

__all__ = [
    "NuclearConsistencyLoss",
    "SpatialCoherenceLoss",
    "BioLossSuite",
    "DAPI_IDX",
    "LAP2_IDX",
    "MARKER_IDX",
    "HAS_KORNIA",
]
