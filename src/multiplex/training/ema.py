"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of model weights as exponential moving average.
EMA weights are used for inference/evaluation, providing smoother and
higher-quality outputs than raw training weights.

Update rule: theta_ema = decay * theta_ema + (1 - decay) * theta_model
Implemented via in-place lerp: shadow.lerp_(param, 1 - decay)

Standard decay for diffusion transformers is 0.9999 (DiT, SiT).

Reference:
    DiT: Scalable Diffusion Models with Transformers
    SiT: Exploring Flow and Diffusion-based Generative Models
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Tracks a shadow copy of all requires_grad parameters via exponential
    moving average. Provides apply/restore for swapping to EMA weights
    during evaluation and state_dict for checkpointing.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor. Default 0.9999 (DiT/SiT standard).
            Higher values = slower update = smoother average.

    Example:
        >>> model = nn.Linear(10, 10)
        >>> ema = EMA(model, decay=0.9999)
        >>> # After each optimizer step:
        >>> ema.update(model)
        >>> # For evaluation:
        >>> ema.apply_shadow(model)
        >>> output = model(input)  # uses EMA weights
        >>> ema.restore(model)     # reverts to training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters toward current model parameters.

        Uses in-place lerp for efficiency (single CUDA kernel per param):
        shadow = decay * shadow + (1 - decay) * param
        Equivalent to: shadow.lerp_(param, 1 - decay)

        Args:
            model: The model with updated parameters.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with EMA shadow parameters.

        Backs up current parameters so they can be restored after
        evaluation via restore().

        Args:
            model: The model to modify in-place.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after apply_shadow.

        Must be called after apply_shadow to revert to training weights.

        Args:
            model: The model to restore in-place.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return copy of shadow parameters for checkpointing.

        Returns:
            Dict mapping parameter names to shadow tensors (cloned).
        """
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load shadow parameters from checkpoint.

        Args:
            state_dict: Dict mapping parameter names to shadow tensors.
        """
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
