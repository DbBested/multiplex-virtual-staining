"""
Training infrastructure for multiplex virtual staining.

This module provides:
- Multi-loss functions for pix2pix-style GAN training
- Configuration dataclasses for training hyperparameters
- Checkpoint management for DDP-safe save/load
- Trainer class for pix2pix-style GAN training
- PatchNCE contrastive loss for misalignment-robust training
- MC Dropout uncertainty estimation for per-pixel confidence maps
- Biological constraint losses for organelle spatial relationships
- Uncertainty-weighted multi-task loss balancing (Kendall et al.)
- JiTTrainer for flow matching training (Phase 21+)
- FlowMatchingLoss, EMA, and ODE samplers for flow matching
"""

from .losses import MultiLoss, DiscriminatorLoss
from .config import TrainConfig
from .checkpoint import CheckpointManager
from .trainer import Trainer
from .patchnce import PatchNCELoss, PatchSampleMLP
from .uncertainty import (
    MCDropoutEstimator,
    enable_dropout,
    disable_dropout,
    compute_spearman_correlation,
    visualize_uncertainty,
)
from .biological_constraints import (
    ExclusionLoss,
    ExclusionConfig,
    ContainmentLoss,
    ContainmentConfig,
    ColocalizationLoss,
    ColocalizationConfig,
    BiologicalConstraintLoss,
    BiologicalConstraintConfig,
)
from .uncertainty_weighting import (
    UncertaintyWeightedLoss,
    UncertaintyWeightConfig,
)
from .curriculum import (
    CurriculumScheduler,
    CurriculumConfig,
)
from .gradient_monitor import GradientConflictMonitor
from .flow_matching import FlowMatchingLoss, euler_sample, heun_sample
from .ema import EMA
from .jit_trainer import JiTTrainer

__all__ = [
    "MultiLoss",
    "DiscriminatorLoss",
    "TrainConfig",
    "CheckpointManager",
    "Trainer",
    "PatchNCELoss",
    "PatchSampleMLP",
    "MCDropoutEstimator",
    "enable_dropout",
    "disable_dropout",
    "compute_spearman_correlation",
    "visualize_uncertainty",
    # Biological constraint losses
    "ExclusionLoss",
    "ExclusionConfig",
    "ContainmentLoss",
    "ContainmentConfig",
    "ColocalizationLoss",
    "ColocalizationConfig",
    "BiologicalConstraintLoss",
    "BiologicalConstraintConfig",
    # Uncertainty weighting
    "UncertaintyWeightedLoss",
    "UncertaintyWeightConfig",
    # Curriculum scheduling
    "CurriculumScheduler",
    "CurriculumConfig",
    # Gradient conflict monitoring
    "GradientConflictMonitor",
    # Flow matching (JiT / Phase 21+)
    "FlowMatchingLoss",
    "euler_sample",
    "heun_sample",
    "EMA",
    "JiTTrainer",
]
