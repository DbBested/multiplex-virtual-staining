"""Baseline method registry for fair comparison.

This module provides a registry pattern for managing baseline image
translation methods, ensuring consistent training and evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.nn as nn


@dataclass
class BaselineConfig:
    """Configuration for a baseline method.

    Attributes:
        name: Baseline method name (e.g., "pix2pix", "cyclegan").
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size for training.
        image_size: Input image size (assumes square).
        extra: Method-specific configuration.
    """
    name: str
    epochs: int = 200
    lr: float = 0.0002
    batch_size: int = 4
    image_size: int = 512
    extra: Dict[str, Any] = field(default_factory=dict)


class BaselineMethod(ABC):
    """Abstract base class for baseline methods.

    All baseline wrappers should inherit from this class and implement
    the required methods for training and inference.
    """

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model: Optional[nn.Module] = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the baseline model architecture."""
        pass

    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute one training step.

        Args:
            batch: Dictionary with 'bf' and 'markers' tensors.

        Returns:
            Dictionary of loss values.
        """
        pass

    @abstractmethod
    def predict(self, bf: torch.Tensor) -> torch.Tensor:
        """Generate marker predictions from brightfield input.

        Args:
            bf: Brightfield images (B, 1, H, W).

        Returns:
            Predicted markers (B, 5, H, W).
        """
        pass

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save model checkpoint with optimizer state."""
        if self.model is None:
            raise RuntimeError("Model not built")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }
        # Save optimizer state if available
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint and return epoch number."""
        if self.model is None:
            self.model = self.build_model()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Restore optimizer state if available
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        return checkpoint.get('epoch', 0)


class BaselineRegistry:
    """Registry for baseline methods.

    Provides factory pattern for creating and managing baseline methods.

    Example:
        >>> registry = BaselineRegistry()
        >>> registry.register("pix2pix", Pix2PixBaseline)
        >>> baseline = registry.get("pix2pix", config)
    """

    _registry: Dict[str, Type[BaselineMethod]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a baseline method.

        Args:
            name: Name for the baseline method.

        Returns:
            Decorator function.

        Example:
            >>> @BaselineRegistry.register("pix2pix")
            ... class Pix2PixBaseline(BaselineMethod):
            ...     pass
        """
        def decorator(method_class: Type[BaselineMethod]) -> Type[BaselineMethod]:
            cls._registry[name] = method_class
            return method_class
        return decorator

    @classmethod
    def get(
        cls,
        name: str,
        config: BaselineConfig,
        device: str = "cuda",
    ) -> BaselineMethod:
        """Get a baseline method by name.

        Args:
            name: Registered baseline name.
            config: Baseline configuration.
            device: Compute device.

        Returns:
            Instantiated baseline method.

        Raises:
            KeyError: If baseline name not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Baseline '{name}' not found. Available: {available}"
            )
        return cls._registry[name](config, device)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered baseline methods."""
        return list(cls._registry.keys())


# Convenience function for registration
def register_baseline(name: str) -> Callable:
    """Register a baseline method with the global registry.

    Shorthand for BaselineRegistry.register().
    """
    return BaselineRegistry.register(name)
