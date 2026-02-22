"""
Checkpoint management for training.

This module provides CheckpointManager for saving and loading training state
with disk quota awareness and DDP-safe operations.

Key features:
- Only rank 0 saves checkpoints (with DDP barrier for sync)
- Automatic cleanup to maintain disk quota
- Handle DDP .module wrapper unwrapping
- Best model tracking based on validation metrics
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


logger = logging.getLogger(__name__)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP or DataParallel wrapper if present.

    Args:
        model: Model potentially wrapped in DDP/DP.

    Returns:
        The underlying model without wrapper.
    """
    if hasattr(model, "module"):
        return model.module
    return model


class CheckpointManager:
    """Manage training checkpoints with disk quota awareness.

    Handles saving/loading of all training state including models,
    optimizers, schedulers, and training progress. Only rank 0 saves
    checkpoints in distributed training.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        keep_n_checkpoints: Number of recent checkpoints to keep.
        rank: Process rank (only rank 0 saves).

    Example:
        >>> manager = CheckpointManager("/tmp/checkpoints", keep_n_checkpoints=3)
        >>> manager.save(generator, discriminator, opt_g, opt_d, sched_g, sched_d,
        ...              epoch=10, step=5000, metrics={"val_loss": 0.5}, config={})
        >>> epoch, step, metrics = manager.load(
        ...     manager.get_latest(), generator, discriminator,
        ...     opt_g, opt_d, sched_g, sched_d, device
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_n_checkpoints: int = 3,
        rank: int = 0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_n_checkpoints = keep_n_checkpoints
        self.rank = rank

        # Best model tracking
        self._best_metric: Optional[float] = None
        self._metric_mode = "min"  # lower is better for loss

        # Create directory only on rank 0
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return dist.is_initialized()

    def _barrier(self) -> None:
        """Synchronize all processes if distributed."""
        if self._is_distributed():
            dist.barrier()

    def save(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g: Optimizer,
        opt_d: Optimizer,
        sched_g: Optional[LRScheduler],
        sched_d: Optional[LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Any,
    ) -> Optional[str]:
        """Save training checkpoint.

        Only saves on rank 0 to avoid duplicate writes. Calls cleanup
        after saving to enforce disk quota.

        Args:
            generator: Generator model (may be DDP wrapped).
            discriminator: Discriminator model (may be DDP wrapped).
            opt_g: Generator optimizer.
            opt_d: Discriminator optimizer.
            sched_g: Generator scheduler (optional).
            sched_d: Discriminator scheduler (optional).
            epoch: Current epoch number.
            step: Current global step.
            metrics: Training/validation metrics dict.
            config: Training configuration (TrainConfig or dict).

        Returns:
            Path to saved checkpoint (only on rank 0), None otherwise.
        """
        if self.rank != 0:
            self._barrier()
            return None

        # Unwrap DDP models
        gen_state = _unwrap_model(generator).state_dict()
        disc_state = _unwrap_model(discriminator).state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "generator_state_dict": gen_state,
            "discriminator_state_dict": disc_state,
            "optimizer_g_state_dict": opt_g.state_dict(),
            "optimizer_d_state_dict": opt_d.state_dict(),
            "scheduler_g_state_dict": sched_g.state_dict() if sched_g else None,
            "scheduler_d_state_dict": sched_d.state_dict() if sched_d else None,
            "metrics": metrics,
            "config": config if isinstance(config, dict) else vars(config) if hasattr(config, "__dict__") else {},
        }

        # Save with epoch number
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")

        # Cleanup old checkpoints
        self.cleanup()

        # Sync with other processes
        self._barrier()

        return str(filepath)

    def load(
        self,
        path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g: Optimizer,
        opt_d: Optimizer,
        sched_g: Optional[LRScheduler],
        sched_d: Optional[LRScheduler],
        device: torch.device,
    ) -> Tuple[int, int, Dict[str, float]]:
        """Load checkpoint and restore training state.

        Args:
            path: Path to checkpoint file.
            generator: Generator model to restore.
            discriminator: Discriminator model to restore.
            opt_g: Generator optimizer to restore.
            opt_d: Discriminator optimizer to restore.
            sched_g: Generator scheduler to restore (optional).
            sched_d: Discriminator scheduler to restore (optional).
            device: Device to map tensors to.

        Returns:
            Tuple of (epoch, step, metrics) from checkpoint.

        Raises:
            FileNotFoundError: If checkpoint path doesn't exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device)

        # Restore model states (handle DDP wrapper)
        _unwrap_model(generator).load_state_dict(checkpoint["generator_state_dict"])
        _unwrap_model(discriminator).load_state_dict(checkpoint["discriminator_state_dict"])

        # Restore optimizer states
        opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        # Restore scheduler states if available
        if sched_g and checkpoint.get("scheduler_g_state_dict"):
            sched_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
        if sched_d and checkpoint.get("scheduler_d_state_dict"):
            sched_d.load_state_dict(checkpoint["scheduler_d_state_dict"])

        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        metrics = checkpoint.get("metrics", {})

        logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, step {step})")

        return epoch, step, metrics

    def get_latest(self) -> Optional[str]:
        """Find most recent checkpoint in directory.

        Returns:
            Path to most recent checkpoint, or None if no checkpoints exist.
        """
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )

        if not checkpoints:
            return None

        return str(checkpoints[0])

    def cleanup(self) -> None:
        """Remove old checkpoints to enforce disk quota.

        Keeps only the most recent keep_n_checkpoints checkpoints.
        """
        if self.rank != 0:
            return

        if not self.checkpoint_dir.exists():
            return

        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Remove oldest checkpoints if over limit
        while len(checkpoints) > self.keep_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")

    def get_disk_usage_mb(self) -> float:
        """Calculate total size of checkpoint directory in MB.

        Logs warning if disk usage exceeds 10GB.

        Returns:
            Total size of checkpoint directory in megabytes.
        """
        if not self.checkpoint_dir.exists():
            return 0.0

        total_size = sum(f.stat().st_size for f in self.checkpoint_dir.glob("*.pt"))
        size_mb = total_size / (1024 * 1024)

        if size_mb > 10 * 1024:  # > 10GB
            logger.warning(f"Checkpoint directory using {size_mb:.1f} MB (> 10GB)")

        return size_mb

    def save_best(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g: Optimizer,
        opt_d: Optimizer,
        sched_g: Optional[LRScheduler],
        sched_d: Optional[LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Any,
        metric_key: str = "val_loss",
        mode: str = "min",
    ) -> Optional[str]:
        """Save as best_model.pt if current metric is better.

        Args:
            generator: Generator model (may be DDP wrapped).
            discriminator: Discriminator model (may be DDP wrapped).
            opt_g: Generator optimizer.
            opt_d: Discriminator optimizer.
            sched_g: Generator scheduler (optional).
            sched_d: Discriminator scheduler (optional).
            epoch: Current epoch number.
            step: Current global step.
            metrics: Training/validation metrics dict.
            config: Training configuration.
            metric_key: Key in metrics dict to compare.
            mode: "min" (lower is better) or "max" (higher is better).

        Returns:
            Path to best_model.pt if saved, None otherwise.
        """
        if self.rank != 0:
            self._barrier()
            return None

        current_metric = metrics.get(metric_key)
        if current_metric is None:
            logger.warning(f"Metric '{metric_key}' not found in metrics dict")
            return None

        # Check if this is the best
        is_best = False
        if self._best_metric is None:
            is_best = True
        elif mode == "min" and current_metric < self._best_metric:
            is_best = True
        elif mode == "max" and current_metric > self._best_metric:
            is_best = True

        if not is_best:
            self._barrier()
            return None

        self._best_metric = current_metric
        self._metric_mode = mode

        # Unwrap DDP models
        gen_state = _unwrap_model(generator).state_dict()
        disc_state = _unwrap_model(discriminator).state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "generator_state_dict": gen_state,
            "discriminator_state_dict": disc_state,
            "optimizer_g_state_dict": opt_g.state_dict(),
            "optimizer_d_state_dict": opt_d.state_dict(),
            "scheduler_g_state_dict": sched_g.state_dict() if sched_g else None,
            "scheduler_d_state_dict": sched_d.state_dict() if sched_d else None,
            "metrics": metrics,
            "config": config if isinstance(config, dict) else vars(config) if hasattr(config, "__dict__") else {},
            "best_metric": self._best_metric,
            "metric_key": metric_key,
            "metric_mode": mode,
        }

        filepath = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, filepath)
        logger.info(f"Saved best model: {filepath} ({metric_key}={current_metric:.6f})")

        self._barrier()

        return str(filepath)
