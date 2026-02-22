"""
Configuration dataclasses for training.

This module provides structured configuration for training using dataclasses
compatible with Hydra and OmegaConf.

The TrainConfig dataclass captures all hyperparameters with pix2pix-standard
defaults suitable for multi-marker virtual staining.
"""

from dataclasses import dataclass
from typing import Tuple

from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Configuration for training.

    This dataclass defines all training hyperparameters with sensible defaults
    following pix2pix conventions.

    Attributes:
        experiment_name: Required name for this experiment (used in logging).
        seed: Random seed for reproducibility.
        max_epochs: Maximum number of training epochs.
        warmup_epochs: Number of epochs for learning rate warmup.
        lr: Base learning rate for both optimizers.
        betas: Adam optimizer beta coefficients.
        lambda_l1: Weight for L1 reconstruction loss.
        lambda_perc: Weight for perceptual (LPIPS) loss.
        lambda_gan: Weight for adversarial loss.
        label_smoothing: Label smoothing for discriminator training.
        batch_size: Batch size per GPU (not total).
        num_workers: Number of DataLoader workers per GPU.
        image_size: Size to resize images to (must be divisible by 32 for U-Net).
        use_bf16: Use bfloat16 mixed precision (requires Ampere+ GPU).
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
        keep_n_checkpoints: Number of recent checkpoints to keep.
        wandb_project: W&B project name for logging.
        log_every_n_steps: Log metrics every N steps.
        sample_every_n_steps: Log sample images every N steps.
        num_samples: Number of sample images to log.
        distributed: Use DistributedDataParallel training.
        use_patchnce: Enable PatchNCE contrastive loss for misalignment-robust training.
        nce_layers: Encoder layer indices to use for PatchNCE.
        nce_t: Temperature for InfoNCE softmax.
        num_patches: Number of patches to sample per layer for PatchNCE.
        lambda_nce: Weight for NCE loss.
        lambda_bio: Weight for biological constraints (0 = disabled).
        bio_weight_exclusion: Weight for exclusion constraint.
        bio_weight_containment: Weight for containment constraint.
        bio_weight_colocalization: Weight for colocalization constraint.
        bio_use_gt_mask: Use GT LMNB1 for nuclear mask during training.
        bio_handle_mitotic: Reduce constraint weight for mitotic cells.
        use_uncertainty_weighting: Enable learned loss weights (Kendall et al.).
        uw_warmup_epochs: Freeze weights for first N epochs before learning.
        uw_min_weight: Minimum bounded weight for uncertainty weighting.
        uw_max_weight: Maximum bounded weight for uncertainty weighting.
        uw_log_every_epoch: Log uncertainty weights every epoch.
        use_cross_marker_attention: Enable cross-marker attention (Phase 12).
        cma_stage1_enabled: Enable Stage 1 bottleneck attention.
        cma_stage1_embed_dim: Embedding dimension for Stage 1 (must match encoder).
        cma_stage1_num_heads: Number of attention heads for Stage 1.
        cma_stage1_dropout: Dropout probability for Stage 1 attention.
        cma_stage2_enabled: Enable Stage 2 output refinement.
        cma_stage2_hidden_dim: Hidden dimension for Stage 2 projections.
        cma_stage2_num_heads: Number of attention heads for Stage 2.
        cma_stage2_dropout: Dropout probability for Stage 2 attention.
        cma_stage2_bypass: Skip Stage 2 at inference for faster prediction.
        use_curriculum: Enable curriculum scheduling for lambda_bio.
        curriculum_warmup_epochs: Epochs with lambda_bio=0 (reconstruction only).
        curriculum_ramp_epochs: Epochs to linearly ramp lambda_bio from 0 to target.
        monitor_gradient_conflicts: Enable gradient conflict monitoring.
        gradient_monitor_every_n_steps: How often to compute gradient conflicts.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.structured(TrainConfig)
        >>> cfg.experiment_name = "baseline_v1"
        >>> print(cfg.lr)
        0.0002
    """

    # Experiment identification
    experiment_name: str = MISSING  # Required field

    # Reproducibility
    seed: int = 42

    # Training schedule
    max_epochs: int = 100
    warmup_epochs: int = 5

    # Optimizer (pix2pix defaults)
    lr: float = 0.0002
    betas: Tuple[float, float] = (0.5, 0.999)

    # Loss weights (pix2pix standard)
    lambda_l1: float = 100.0
    lambda_perc: float = 10.0
    lambda_gan: float = 1.0
    label_smoothing: float = 0.0
    per_marker_lpips: bool = False  # Compute LPIPS per marker for better perceptual quality

    # Data loading
    batch_size: int = 4  # per GPU
    num_workers: int = 4
    image_size: int = 512  # Resize images to this size (must be divisible by 32)

    # Mixed precision
    use_bf16: bool = True

    # Checkpointing
    checkpoint_dir: str = "${oc.env:SCRATCH,/tmp}/checkpoints"
    checkpoint_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3

    # Logging
    wandb_project: str = "multiplex-virtual-ihc"
    log_every_n_steps: int = 50
    sample_every_n_steps: int = 500
    num_samples: int = 4

    # Distributed training
    distributed: bool = True

    # PatchNCE loss (for pseudo-pair/misalignment-robust training)
    use_patchnce: bool = False  # Enable PatchNCE loss
    nce_layers: Tuple[int, ...] = (0, 2)  # Encoder layers to use
    nce_t: float = 0.07  # Temperature for InfoNCE
    num_patches: int = 256  # Patches to sample per layer
    lambda_nce: float = 1.0  # Weight for NCE loss

    # Biological constraint loss (v2.0)
    # Penalizes violations of known organelle spatial relationships
    lambda_bio: float = 0.0  # Weight for biological constraints (0 = disabled)
    bio_weight_exclusion: float = 1.0  # Weight for exclusion constraint
    bio_weight_containment: float = 1.0  # Weight for containment constraint (affects FBL)
    bio_weight_colocalization: float = 1.0  # Weight for colocalization constraint
    bio_use_gt_mask: bool = True  # Use GT LMNB1 for nuclear mask during training
    bio_handle_mitotic: bool = True  # Reduce constraint weight for mitotic cells
    bio_enable_containment: bool = True  # Enable containment loss (set False to not affect FBL)

    # Uncertainty weighting (Phase 11)
    # Learns optimal weights for each loss component via Kendall et al. method
    use_uncertainty_weighting: bool = False  # Enable learned loss weights
    uw_warmup_epochs: int = 20  # Freeze weights for first N epochs
    uw_min_weight: float = 0.01  # Minimum bounded weight
    uw_max_weight: float = 10.0  # Maximum bounded weight
    uw_log_every_epoch: bool = True  # Log weights every epoch (not step)

    # Cross-marker attention (Phase 12)
    # Enables inter-marker relationship modeling via attention
    use_cross_marker_attention: bool = False  # Enable cross-marker attention
    cma_stage1_enabled: bool = True  # Enable Stage 1 (bottleneck attention)
    cma_stage1_embed_dim: int = 1024  # Must match encoder bottleneck
    cma_stage1_num_heads: int = 8  # Number of attention heads for Stage 1
    cma_stage1_dropout: float = 0.1  # Dropout probability for Stage 1
    cma_stage2_enabled: bool = True  # Enable Stage 2 (output refinement)
    cma_stage2_hidden_dim: int = 64  # Hidden dimension for Stage 2 projections
    cma_stage2_num_heads: int = 4  # Number of attention heads for Stage 2
    cma_stage2_dropout: float = 0.1  # Dropout probability for Stage 2
    cma_stage2_bypass: bool = False  # Skip Stage 2 at inference

    # Curriculum scheduling (Phase 13)
    # Gradually ramps biological constraint weight over training
    use_curriculum: bool = False  # Enable curriculum scheduling for lambda_bio
    curriculum_warmup_epochs: int = 20  # Epochs with lambda_bio=0
    curriculum_ramp_epochs: int = 30  # Epochs to linearly ramp to target

    # Gradient conflict monitoring (Phase 13)
    # Tracks cosine similarity between reconstruction and constraint gradients
    monitor_gradient_conflicts: bool = False  # Enable gradient monitoring
    gradient_monitor_every_n_steps: int = 100  # How often to compute conflicts
