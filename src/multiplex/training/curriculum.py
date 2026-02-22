"""
Curriculum scheduling for biological constraint ramp-up.

This module provides:
- CurriculumConfig: Configuration for curriculum-based loss scheduling
- CurriculumScheduler: Scheduler for gradual constraint introduction

The curriculum schedule enables gradual introduction of biological constraints
during training, allowing reconstruction loss to stabilize before constraint
losses influence gradients.

Schedule:
- Warmup phase: Epochs [0, warmup_epochs) - lambda_bio = 0
- Ramp phase: Epochs [warmup_epochs, warmup_epochs + ramp_epochs) - linear ramp
- Full phase: Epochs [warmup_epochs + ramp_epochs, ...) - lambda_bio = target
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based loss scheduling.

    Attributes:
        warmup_epochs: Epochs with reconstruction loss only (no constraints).
        ramp_epochs: Epochs to linearly ramp constraint weight from 0 to target.
        target_lambda_bio: Final lambda_bio weight after ramp completes.

    The schedule is:
    - Epochs [0, warmup_epochs): lambda_bio = 0
    - Epochs [warmup_epochs, warmup_epochs + ramp_epochs): linear ramp
    - Epochs [warmup_epochs + ramp_epochs, ...): lambda_bio = target

    Example:
        >>> config = CurriculumConfig(warmup_epochs=20, ramp_epochs=30, target_lambda_bio=0.1)
        >>> config.warmup_epochs
        20
    """

    warmup_epochs: int = 20
    ramp_epochs: int = 30
    target_lambda_bio: float = 0.1


class CurriculumScheduler:
    """Curriculum scheduler for biological constraint ramp-up.

    Implements linear ramp of lambda_bio after initial warmup period.
    Called at the start of each epoch to get current lambda_bio.

    Example:
        >>> scheduler = CurriculumScheduler(CurriculumConfig())
        >>> scheduler.get_lambda_bio(epoch=10)  # warmup: returns 0.0
        0.0
        >>> scheduler.get_lambda_bio(epoch=35)  # mid-ramp: returns ~0.05
        0.05
        >>> scheduler.get_lambda_bio(epoch=60)  # post-ramp: returns 0.1
        0.1
    """

    def __init__(self, config: CurriculumConfig):
        """Initialize the curriculum scheduler.

        Args:
            config: Curriculum configuration with warmup, ramp, and target settings.
        """
        self.config = config

    def get_lambda_bio(self, epoch: int) -> float:
        """Get current lambda_bio based on epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Current lambda_bio weight.
        """
        cfg = self.config

        if epoch < cfg.warmup_epochs:
            # Pure reconstruction phase
            return 0.0

        ramp_progress = epoch - cfg.warmup_epochs
        if ramp_progress >= cfg.ramp_epochs:
            # Full constraint weight
            return cfg.target_lambda_bio

        # Linear ramp: 0 -> target over ramp_epochs
        ramp_fraction = ramp_progress / cfg.ramp_epochs
        return cfg.target_lambda_bio * ramp_fraction

    def get_schedule_state(self, epoch: int) -> Dict[str, float]:
        """Get full schedule state for logging.

        Returns dict suitable for W&B logging with curriculum/ prefix.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Dict with curriculum/lambda_bio, curriculum/epoch,
            curriculum/phase, and curriculum/ramp_progress.
        """
        lambda_bio = self.get_lambda_bio(epoch)
        phase_name = self._get_phase_name(epoch)

        # Convert phase name to numeric for logging
        phase_to_num = {"warmup": 0, "ramp": 1, "full": 2}
        phase_num = phase_to_num.get(phase_name, 0)

        return {
            "curriculum/lambda_bio": lambda_bio,
            "curriculum/epoch": float(epoch),
            "curriculum/phase": float(phase_num),
            "curriculum/ramp_progress": self._get_ramp_progress(epoch),
        }

    def _get_phase_name(self, epoch: int) -> str:
        """Get the current phase name.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Phase name: "warmup", "ramp", or "full".
        """
        cfg = self.config
        if epoch < cfg.warmup_epochs:
            return "warmup"
        elif epoch < cfg.warmup_epochs + cfg.ramp_epochs:
            return "ramp"
        else:
            return "full"

    def _get_ramp_progress(self, epoch: int) -> float:
        """Get the ramp progress from 0.0 to 1.0.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Ramp progress: 0.0 during warmup, 0.0-1.0 during ramp, 1.0 after.
        """
        cfg = self.config
        if epoch < cfg.warmup_epochs:
            return 0.0
        ramp_progress = epoch - cfg.warmup_epochs
        return min(1.0, ramp_progress / cfg.ramp_epochs)
