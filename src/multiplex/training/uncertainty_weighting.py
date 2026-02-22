"""
Uncertainty-weighted multi-task loss balancing (Kendall et al.).

This module provides learned loss weighting for automatic multi-task loss balancing
using homoscedastic uncertainty, as described in "Multi-Task Learning Using
Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al., CVPR 2018).

The key insight is that task-specific uncertainty (sigma) can be used to weight losses:
- Higher sigma -> lower weight (down-weight uncertain tasks)
- Regularization term (log sigma) prevents weights from collapsing to zero

This implementation includes:
- Log-sigma parameterization for numerical stability
- Sigmoid bounding to keep weights in [0.01, 10.0] range
- Warm-up period to freeze weights during initial training
- Per-task weights for L1, perceptual, GAN, exclusion, containment, colocalization

Example:
    >>> from multiplex.training.uncertainty_weighting import (
    ...     UncertaintyWeightedLoss, UncertaintyWeightConfig
    ... )
    >>> import torch
    >>> config = UncertaintyWeightConfig(num_tasks=6, warmup_epochs=20)
    >>> uw_loss = UncertaintyWeightedLoss(config)
    >>> losses = [torch.tensor(0.5) for _ in range(6)]
    >>> total, loss_dict = uw_loss(losses)
    >>> print(f"Total weighted loss: {total.item():.4f}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class UncertaintyWeightConfig:
    """Configuration for uncertainty-weighted loss.

    Attributes:
        num_tasks: Number of loss components to weight. Default 6
            (L1, perceptual, GAN, exclusion, containment, colocalization).
        warmup_epochs: Number of epochs to freeze weights before learning. Default 20.
        min_weight: Minimum bounded weight (via sigmoid). Default 0.01.
        max_weight: Maximum bounded weight (via sigmoid). Default 10.0.
        init_log_sigma: Optional custom initialization for log_sigma per task.
            If None, initializes all to 0 (sigma=1, weight~0.5).
        eps: Small constant for numerical stability. Default 1e-8.

    Example:
        >>> config = UncertaintyWeightConfig(num_tasks=6, warmup_epochs=20)
        >>> config.min_weight
        0.01
        >>> config.max_weight
        10.0
    """

    num_tasks: int = 6
    warmup_epochs: int = 20
    min_weight: float = 0.01
    max_weight: float = 10.0
    init_log_sigma: Optional[List[float]] = None
    eps: float = 1e-8


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss with Kendall regularization.

    Learns optimal weights for each loss component using homoscedastic
    uncertainty. Higher uncertainty (sigma) decreases weight, but the
    regularization term (log sigma) prevents collapse to zero.

    The Kendall formula for task i:
        weighted_loss_i = weight_i * loss_i + log_sigma_i

    Where weight = 0.5 * exp(-2 * log_sigma), bounded via sigmoid to [min, max].

    The log_sigma parameters are frozen during a warm-up period to allow
    reconstruction loss to stabilize before weight learning begins.

    Args:
        config: UncertaintyWeightConfig instance. Uses defaults if None.

    Attributes:
        TASK_NAMES: Class attribute mapping task indices to names for logging.
        log_sigma: Learnable nn.Parameter of shape (num_tasks,).

    Example:
        >>> config = UncertaintyWeightConfig(num_tasks=6, warmup_epochs=20)
        >>> uw_loss = UncertaintyWeightedLoss(config)
        >>> # Check initial frozen state
        >>> print(f"Frozen: {uw_loss._is_frozen}")
        Frozen: True
        >>> # Get weights before unfreeze
        >>> weights = uw_loss.get_effective_weights()
        >>> print(f"Initial weights: {weights}")

    Example with forward pass:
        >>> losses = [torch.tensor(0.5) for _ in range(6)]
        >>> total, loss_dict = uw_loss(losses, return_weights=True)
        >>> print(f"Total: {total.item():.4f}")
        >>> print(f"Keys: {list(loss_dict.keys())}")
    """

    # Task names for logging (matches loss component order)
    TASK_NAMES = ['l1', 'perc', 'gan', 'excl', 'contain', 'coloc']

    def __init__(self, config: Optional[UncertaintyWeightConfig] = None):
        super().__init__()
        self.config = config or UncertaintyWeightConfig()

        # Initialize log_sigma parameters
        if self.config.init_log_sigma is not None:
            if len(self.config.init_log_sigma) != self.config.num_tasks:
                raise ValueError(
                    f"init_log_sigma length ({len(self.config.init_log_sigma)}) "
                    f"must match num_tasks ({self.config.num_tasks})"
                )
            init_vals = torch.tensor(self.config.init_log_sigma, dtype=torch.float32)
        else:
            init_vals = torch.zeros(self.config.num_tasks)

        self.log_sigma = nn.Parameter(init_vals)

        # Start frozen for warm-up period
        self.log_sigma.requires_grad = False
        self._is_frozen = True

    def maybe_unfreeze(self, current_epoch: int) -> bool:
        """Unfreeze weights after warm-up period.

        Should be called at the start of each epoch during training.
        After warmup_epochs, enables gradient computation for log_sigma
        so the optimizer can learn optimal task weights.

        Args:
            current_epoch: Current training epoch (0-indexed).

        Returns:
            True if unfreezing occurred this call, False otherwise.
            Only returns True once - subsequent calls return False.

        Example:
            >>> uw_loss = UncertaintyWeightedLoss()
            >>> # During warm-up
            >>> uw_loss.maybe_unfreeze(10)  # Returns False
            False
            >>> # At warm-up boundary
            >>> uw_loss.maybe_unfreeze(20)  # Returns True
            True
            >>> # Already unfrozen
            >>> uw_loss.maybe_unfreeze(21)  # Returns False
            False
        """
        if current_epoch >= self.config.warmup_epochs and self._is_frozen:
            self.log_sigma.requires_grad = True
            self._is_frozen = False
            return True
        return False

    def get_effective_weights(self) -> torch.Tensor:
        """Compute bounded effective weights from log_sigma.

        Converts the learned log_sigma parameters to effective weights
        using the Kendall formula with sigmoid bounding:

        1. Compute raw precision: 0.5 * exp(-2 * log_sigma)
        2. Apply sigmoid to log(precision) for smooth bounding
        3. Scale sigmoid output to [min_weight, max_weight]

        Returns:
            Tensor of shape (num_tasks,) with weights in [min_weight, max_weight].

        Example:
            >>> uw_loss = UncertaintyWeightedLoss()
            >>> weights = uw_loss.get_effective_weights()
            >>> # At initialization (log_sigma=0), weights are ~0.5
            >>> print(f"Weights: {weights}")
            >>> # All weights should be in bounds
            >>> assert (weights >= 0.01).all()
            >>> assert (weights <= 10.0).all()
        """
        cfg = self.config

        # Raw precision: 1 / (2 * sigma^2) = 0.5 * exp(-2 * log_sigma)
        raw_precision = 0.5 * torch.exp(-2 * self.log_sigma)

        # Apply sigmoid bounding for smooth, differentiable bounds
        # Map through sigmoid then scale to [min, max]
        # Use log(precision) as sigmoid input for stable transition
        log_precision = torch.log(raw_precision + cfg.eps)
        normalized = torch.sigmoid(log_precision)
        bounded = cfg.min_weight + (cfg.max_weight - cfg.min_weight) * normalized

        return bounded

    def forward(
        self,
        losses: List[torch.Tensor],
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute uncertainty-weighted total loss.

        Applies the Kendall formula to weight each loss component:
            weighted_loss = sum_i (weight_i * loss_i + log_sigma_i)

        The regularization term (log_sigma_i) is critical - it penalizes
        low weights (high sigma) and prevents collapse to zero.

        Args:
            losses: List of individual loss tensors (scalars), one per task.
                Must have length equal to config.num_tasks.
            return_weights: If True, include effective weights and log_sigma
                values in the returned dictionary.

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Scalar tensor with weighted sum of all losses.
            - loss_dict: Dictionary with individual loss values:
                - loss_{task}_raw: Raw (unweighted) loss value
                - loss_{task}_weighted: Weight * loss value
                - loss_uw_total: Total weighted loss
                - weight_{task}: Effective weight (if return_weights=True)
                - log_sigma_{task}: Log sigma value (if return_weights=True)

        Raises:
            ValueError: If len(losses) != config.num_tasks.

        Example:
            >>> uw_loss = UncertaintyWeightedLoss()
            >>> losses = [torch.tensor(0.5 * (i + 1)) for i in range(6)]
            >>> total, loss_dict = uw_loss(losses, return_weights=True)
            >>> print(f"Total: {total.item():.4f}")
            >>> print(f"L1 raw: {loss_dict['loss_l1_raw']:.4f}")
            >>> print(f"L1 weight: {loss_dict['weight_l1']:.4f}")
        """
        cfg = self.config

        if len(losses) != cfg.num_tasks:
            raise ValueError(
                f"Expected {cfg.num_tasks} losses, got {len(losses)}"
            )

        # Get device from first loss tensor
        device = losses[0].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, float] = {}

        effective_weights = self.get_effective_weights()

        for i, loss in enumerate(losses):
            # Kendall formulation with bounded weights
            # weighted_loss = weight * loss + regularization (log_sigma)
            weight = effective_weights[i]

            # Regularization: log(sigma) = log_sigma (direct from parameter)
            reg = self.log_sigma[i]

            weighted_loss = weight * loss + reg
            total_loss = total_loss + weighted_loss

            # Log individual components
            task_name = self.TASK_NAMES[i] if i < len(self.TASK_NAMES) else f"task_{i}"
            loss_dict[f"loss_{task_name}_raw"] = loss.item()
            loss_dict[f"loss_{task_name}_weighted"] = (weight * loss).item()

        loss_dict["loss_uw_total"] = total_loss.item()

        # Optionally include weights for logging
        if return_weights:
            for i, weight in enumerate(effective_weights):
                task_name = self.TASK_NAMES[i] if i < len(self.TASK_NAMES) else f"task_{i}"
                loss_dict[f"weight_{task_name}"] = weight.item()
                loss_dict[f"log_sigma_{task_name}"] = self.log_sigma[i].item()

        return total_loss, loss_dict

    def get_weight_dict(self) -> Dict[str, float]:
        """Get current weights for W&B logging.

        Returns a dictionary with all uncertainty weighting metrics
        suitable for logging to W&B at epoch level.

        Returns:
            Dictionary with keys:
            - uw/log_sigma_{task}: Raw log_sigma parameter value
            - uw/weight_{task}: Effective bounded weight
            - uw/sigma_{task}: Sigma value (exp(log_sigma))
            - uw/is_frozen: 1.0 if frozen, 0.0 if learning

        Example:
            >>> uw_loss = UncertaintyWeightedLoss()
            >>> weight_dict = uw_loss.get_weight_dict()
            >>> print(f"L1 weight: {weight_dict['uw/weight_l1']:.4f}")
            >>> print(f"Frozen: {weight_dict['uw/is_frozen']}")
        """
        weights = self.get_effective_weights()
        result: Dict[str, float] = {}

        for i in range(self.config.num_tasks):
            task_name = self.TASK_NAMES[i] if i < len(self.TASK_NAMES) else f"task_{i}"
            result[f"uw/log_sigma_{task_name}"] = self.log_sigma[i].item()
            result[f"uw/weight_{task_name}"] = weights[i].item()
            result[f"uw/sigma_{task_name}"] = torch.exp(self.log_sigma[i]).item()

        result["uw/is_frozen"] = float(self._is_frozen)

        return result


# Convenience exports
__all__ = [
    "UncertaintyWeightedLoss",
    "UncertaintyWeightConfig",
]
