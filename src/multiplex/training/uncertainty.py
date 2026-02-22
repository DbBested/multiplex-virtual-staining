"""
MC Dropout uncertainty estimation for virtual staining predictions.

This module provides Monte Carlo Dropout-based uncertainty quantification,
enabling per-pixel confidence maps for predicted marker images.

Key components:
- enable_dropout: Toggle dropout layers to train mode during inference
- MCDropoutEstimator: Run multiple forward passes and compute statistics
- compute_spearman_correlation: Validate uncertainty-error correlation

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during eval mode for MC Dropout inference.

    Sets all Dropout and Dropout2d modules to training mode while keeping
    BatchNorm and other layers in eval mode.

    Args:
        model: PyTorch model with Dropout/Dropout2d layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def disable_dropout(model: nn.Module) -> None:
    """Disable dropout layers (return to standard eval mode).

    Args:
        model: PyTorch model with Dropout/Dropout2d layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.eval()


class MCDropoutEstimator:
    """Monte Carlo Dropout uncertainty estimator.

    Runs multiple stochastic forward passes with dropout enabled and computes
    per-pixel mean and standard deviation as uncertainty estimate.

    Args:
        num_samples: Number of forward passes. Default 20.
            Higher values give more stable estimates but slower inference.
        reduce_channels: Whether to average uncertainty across channels.
            Default True returns (B, 1, H, W), False returns (B, C, H, W).

    Example:
        >>> estimator = MCDropoutEstimator(num_samples=20)
        >>> model = AttentionUNetGenerator(dropout_p=0.2)
        >>> bf = torch.randn(1, 1, 512, 512)
        >>> mean, uncertainty = estimator(model, bf)
        >>> # uncertainty is high where model is unsure
    """

    def __init__(
        self,
        num_samples: int = 20,
        reduce_channels: bool = True,
    ):
        self.num_samples = num_samples
        self.reduce_channels = reduce_channels

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MC Dropout inference.

        Args:
            model: Generator model with Dropout2d layers.
            x: Input tensor (B, C, H, W).

        Returns:
            Tuple of:
            - mean: Mean prediction (B, num_markers, H, W)
            - uncertainty: Standard deviation / uncertainty
              Shape (B, 1, H, W) if reduce_channels else (B, num_markers, H, W)
        """
        model.eval()
        enable_dropout(model)

        predictions = []
        for _ in range(self.num_samples):
            pred = model(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (T, B, C, H, W)
        mean = predictions.mean(dim=0)  # (B, C, H, W)
        std = predictions.std(dim=0)  # (B, C, H, W)

        if self.reduce_channels:
            # Average uncertainty across marker channels
            uncertainty = std.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            uncertainty = std

        # Restore eval mode
        disable_dropout(model)

        return mean, uncertainty

    def get_sample_predictions(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Get all MC Dropout samples (useful for visualization).

        Args:
            model: Generator model with Dropout2d layers.
            x: Input tensor (B, C, H, W).

        Returns:
            All predictions (num_samples, B, num_markers, H, W)
        """
        model.eval()
        enable_dropout(model)

        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = model(x)
                predictions.append(pred)

        disable_dropout(model)
        return torch.stack(predictions, dim=0)


def compute_spearman_correlation(
    uncertainty: torch.Tensor,
    error: torch.Tensor,
) -> float:
    """Compute Spearman correlation between uncertainty and prediction error.

    Used to validate that uncertainty estimates are meaningful - high uncertainty
    should correlate with high prediction error.

    Args:
        uncertainty: Uncertainty map (B, 1, H, W) or (B, C, H, W).
        error: Absolute error map (B, C, H, W).

    Returns:
        Spearman correlation coefficient (float). Target: > 0.3 for good calibration.

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError("scipy required for Spearman correlation: pip install scipy")

    # If uncertainty has fewer channels than error, expand it
    # uncertainty: (B, 1, H, W) -> expand to (B, C, H, W) for per-pixel comparison
    if uncertainty.dim() == 4 and error.dim() == 4:
        if uncertainty.shape[1] < error.shape[1]:
            uncertainty = uncertainty.expand(-1, error.shape[1], -1, -1)
        elif error.shape[1] < uncertainty.shape[1]:
            error = error.expand(-1, uncertainty.shape[1], -1, -1)

    # Flatten tensors
    unc_flat = uncertainty.flatten().cpu().numpy()
    err_flat = error.flatten().cpu().numpy()

    # Handle edge case of zero variance
    if np.std(unc_flat) < 1e-8 or np.std(err_flat) < 1e-8:
        return 0.0

    corr, p_value = spearmanr(unc_flat, err_flat)

    return float(corr) if not np.isnan(corr) else 0.0


def visualize_uncertainty(
    uncertainty: torch.Tensor,
    normalize: bool = True,
    colormap: str = "viridis",
) -> np.ndarray:
    """Convert uncertainty tensor to visualization-ready numpy array.

    Args:
        uncertainty: Uncertainty map (H, W) or (1, H, W) or (B, 1, H, W).
        normalize: Whether to normalize to [0, 1]. Default True.
        colormap: Matplotlib colormap name. Default "viridis".

    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255].
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Return grayscale if matplotlib not available
        unc = uncertainty.squeeze().cpu().numpy()
        if normalize:
            unc = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)
        return (unc * 255).astype(np.uint8)

    unc = uncertainty.squeeze().cpu().numpy()
    if normalize:
        unc = (unc - unc.min()) / (unc.max() - unc.min() + 1e-8)

    cmap = plt.get_cmap(colormap)
    colored = cmap(unc)[:, :, :3]  # Remove alpha channel
    return (colored * 255).astype(np.uint8)
