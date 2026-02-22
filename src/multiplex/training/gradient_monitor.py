"""
Gradient conflict monitoring for multi-task learning.

This module provides GradientConflictMonitor to track cosine similarity
between gradients of different loss components (e.g., reconstruction vs
biological constraints). Negative cosine similarity indicates conflicting
optimization directions.

Based on PCGrad paper (NeurIPS 2020) gradient conflict detection.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class GradientConflictMonitor:
    """Monitor gradient conflicts between loss components.

    Computes cosine similarity between gradients of different losses
    to detect destructive interference. Negative similarity indicates
    conflicting gradients.

    Usage:
        1. Compute loss components separately
        2. Call monitor.compute_conflicts() with losses and model
        3. Log conflict metrics to W&B

    Args:
        loss_names: Names of loss components to monitor. Default ['recon', 'bio'].
        log_every_n_steps: How often to compute conflicts. Default 100.

    Example:
        >>> monitor = GradientConflictMonitor()
        >>> losses = {'recon': loss_recon, 'bio': loss_bio}
        >>> conflicts = monitor.compute_conflicts(losses, model)
        >>> if conflicts:
        ...     print(conflicts['grad_conflict/cos_sim_recon_bio'])
    """

    def __init__(
        self,
        loss_names: Optional[List[str]] = None,
        log_every_n_steps: int = 100,
    ):
        self.loss_names = loss_names or ['recon', 'bio']
        self.log_every_n_steps = log_every_n_steps
        self._step_count = 0
        self._history: List[Dict[str, float]] = []

    def compute_conflicts(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        compute_full: bool = False,
    ) -> Optional[Dict[str, float]]:
        """Compute gradient conflicts between loss components.

        Args:
            losses: Dict mapping loss name to scalar loss tensor.
            model: Model to compute gradients for.
            compute_full: If True, always compute. If False, only every N steps.

        Returns:
            Dict with cosine similarity metrics, or None if skipped.
        """
        self._step_count += 1

        if not compute_full and self._step_count % self.log_every_n_steps != 0:
            return None

        # Get shared parameters (typically generator parameters)
        params = list(model.parameters())

        # Compute gradients for each loss component
        grads: Dict[str, torch.Tensor] = {}
        for name, loss in losses.items():
            if loss is None:
                continue
            if not isinstance(loss, torch.Tensor):
                continue
            if not loss.requires_grad:
                # Handle case where loss doesn't require grad (e.g., zero tensor)
                continue

            # Compute gradient without accumulating
            grad_list = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            # Flatten all gradients into single vector
            grad_flat = torch.cat([
                g.flatten() if g is not None else torch.zeros_like(p).flatten()
                for g, p in zip(grad_list, params)
            ])
            grads[name] = grad_flat

        # If we don't have at least 2 gradients, nothing to compare
        if len(grads) < 2:
            return None

        # Compute pairwise cosine similarities
        conflicts: Dict[str, float] = {}
        names = list(grads.keys())
        for i, name_i in enumerate(names):
            for name_j in names[i + 1:]:
                cos_sim = self._cosine_similarity(grads[name_i], grads[name_j])
                key = f"grad_conflict/cos_sim_{name_i}_{name_j}"
                conflicts[key] = cos_sim

                # Also track if conflicting (negative)
                conflicts[f"grad_conflict/is_conflict_{name_i}_{name_j}"] = float(cos_sim < 0)

        # Add gradient magnitudes for context
        for name, grad in grads.items():
            conflicts[f"grad_conflict/norm_{name}"] = grad.norm().item()

        self._history.append(conflicts)
        return conflicts

    def _cosine_similarity(self, g1: torch.Tensor, g2: torch.Tensor) -> float:
        """Compute cosine similarity between two gradient vectors.

        Args:
            g1: First gradient vector.
            g2: Second gradient vector.

        Returns:
            Cosine similarity in [-1, 1]. Negative indicates conflict.
        """
        norm1 = g1.norm()
        norm2 = g2.norm()

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return (g1 @ g2 / (norm1 * norm2)).item()

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics over history.

        Returns:
            Dict with mean/min cosine similarity and conflict rate.
            Empty dict if no history.
        """
        if not self._history:
            return {}

        # Collect all cosine similarity values
        all_cos_sims = []
        for record in self._history:
            for key, value in record.items():
                if 'cos_sim' in key:
                    all_cos_sims.append(value)

        if not all_cos_sims:
            return {}

        return {
            "grad_conflict/mean_cos_sim": float(np.mean(all_cos_sims)),
            "grad_conflict/min_cos_sim": float(np.min(all_cos_sims)),
            "grad_conflict/conflict_rate": float(np.mean([s < 0 for s in all_cos_sims])),
        }

    def reset_history(self) -> None:
        """Clear the history buffer."""
        self._history.clear()
