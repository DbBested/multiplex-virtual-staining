"""
Trainer class for pix2pix-style GAN training.

This module provides the core Trainer class for training the virtual staining
model with mixed precision support and DDP-compatible training loop.

Key features:
- Alternating generator/discriminator updates (pix2pix style)
- BF16 mixed precision on supported hardware (Ampere+)
- DDP-aware training with proper sampler synchronization
- Modular training_step/validation_step for testing
- W&B experiment tracking with loss curves and sample visualizations
"""

import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OmegaConf = None
    OMEGACONF_AVAILABLE = False

from multiplex.training.config import TrainConfig
from multiplex.training.losses import DiscriminatorLoss, MultiLoss
from multiplex.training.biological_constraints import BiologicalConstraintConfig
from multiplex.training.checkpoint import CheckpointManager
from multiplex.training.uncertainty_weighting import (
    UncertaintyWeightedLoss,
    UncertaintyWeightConfig,
)
from multiplex.training.curriculum import CurriculumScheduler, CurriculumConfig
from multiplex.training.gradient_monitor import GradientConflictMonitor


logger = logging.getLogger(__name__)


class LinearWarmupScheduler:
    """Linear learning rate warmup wrapper.

    Wraps a scheduler to add linear warmup for the first N epochs.
    During warmup, learning rate increases linearly from 0 to base_lr.

    Args:
        optimizer: Optimizer to wrap.
        base_scheduler: Scheduler to use after warmup.
        warmup_epochs: Number of epochs for warmup.
        last_epoch: Last epoch number (for resuming).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: LRScheduler,
        warmup_epochs: int,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self._step_count = last_epoch + 1 if last_epoch >= 0 else 0

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        """Update learning rate for next epoch."""
        self._step_count += 1

        if self._step_count <= self.warmup_epochs:
            # Linear warmup: lr = base_lr * (step / warmup_epochs)
            warmup_factor = self._step_count / self.warmup_epochs
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group["lr"] = base_lr * warmup_factor
        else:
            # Use base scheduler after warmup
            self.base_scheduler.step()

    def get_last_lr(self) -> list:
        """Get current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            "step_count": self._step_count,
            "base_scheduler": self.base_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self._step_count = state_dict["step_count"]
        self.base_scheduler.load_state_dict(state_dict["base_scheduler"])


def _get_autocast_dtype() -> torch.dtype:
    """Get the best autocast dtype for current hardware.

    Returns BF16 if supported (Ampere+), otherwise FP16.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class Trainer:
    """Main trainer for pix2pix-style GAN training.

    Handles alternating generator/discriminator updates with mixed precision
    training and DDP support.

    Args:
        generator: AttentionUNetGenerator model.
        discriminator: PatchGAN70 model.
        config: TrainConfig with all hyperparameters.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Training device (cuda or cpu).
        rank: DDP rank (0 = main process).
        world_size: Total number of DDP processes.

    Example:
        >>> trainer = Trainer(generator, discriminator, config,
        ...                   train_loader, val_loader, device)
        >>> trainer.fit()
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # Set random seeds
        self._set_seed(config.seed)

        # Setup losses
        # Create biological constraint config if enabled
        bio_config = None
        if getattr(config, 'lambda_bio', 0) > 0:
            bio_config = BiologicalConstraintConfig(
                weight_exclusion=getattr(config, 'bio_weight_exclusion', 1.0),
                weight_containment=getattr(config, 'bio_weight_containment', 1.0),
                weight_colocalization=getattr(config, 'bio_weight_colocalization', 1.0),
                use_gt_mask=getattr(config, 'bio_use_gt_mask', True),
                handle_mitotic=getattr(config, 'bio_handle_mitotic', True),
                enable_containment=getattr(config, 'bio_enable_containment', True),
            )

        self.multi_loss = MultiLoss(
            lambda_l1=config.lambda_l1,
            lambda_perc=config.lambda_perc,
            lambda_gan=config.lambda_gan,
            lambda_bio=getattr(config, 'lambda_bio', 0.0),
            bio_config=bio_config,
            device=str(device),
            per_marker_lpips=getattr(config, 'per_marker_lpips', False),
        ).to(device)

        self.disc_loss = DiscriminatorLoss(
            label_smoothing=config.label_smoothing
        )

        # Uncertainty weighting (optional, Phase 11)
        self.uw_loss = None
        if getattr(config, 'use_uncertainty_weighting', False):
            uw_config = UncertaintyWeightConfig(
                num_tasks=6,  # L1, perc, GAN, excl, contain, coloc
                warmup_epochs=config.uw_warmup_epochs,
                min_weight=config.uw_min_weight,
                max_weight=config.uw_max_weight,
            )
            self.uw_loss = UncertaintyWeightedLoss(uw_config).to(device)

        # Curriculum scheduler (optional, Phase 13)
        self.curriculum_scheduler = None
        if getattr(config, 'use_curriculum', False):
            curriculum_config = CurriculumConfig(
                warmup_epochs=config.curriculum_warmup_epochs,
                ramp_epochs=config.curriculum_ramp_epochs,
                target_lambda_bio=config.lambda_bio,  # Use config lambda_bio as target
            )
            self.curriculum_scheduler = CurriculumScheduler(curriculum_config)

        # Gradient conflict monitor (optional, Phase 13)
        self.gradient_monitor = None
        if getattr(config, 'monitor_gradient_conflicts', False):
            self.gradient_monitor = GradientConflictMonitor(
                log_every_n_steps=config.gradient_monitor_every_n_steps,
            )

        # Setup optimizers and schedulers
        self.opt_g, self.opt_d, self.sched_g, self.sched_d = self._setup_optimizers()

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_n_checkpoints=config.keep_n_checkpoints,
            rank=rank,
        )

        # Mixed precision setup
        self.use_bf16 = config.use_bf16
        self.autocast_dtype = _get_autocast_dtype() if config.use_bf16 else torch.float32

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.start_epoch = 0  # For resume support

        # W&B tracking
        self.wandb_run = None
        self._init_wandb()

        logger.info(
            f"Trainer initialized: rank={rank}, world_size={world_size}, "
            f"autocast_dtype={self.autocast_dtype}, device={device}"
        )

    def _set_seed(self, seed: int) -> None:
        """Set all random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        random.seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        torch.manual_seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)

    def _setup_optimizers(self) -> Tuple[Adam, Adam, LinearWarmupScheduler, LinearWarmupScheduler]:
        """Setup optimizers and learning rate schedulers.

        Returns:
            Tuple of (opt_g, opt_d, sched_g, sched_d).
        """
        # Build generator parameter groups
        # If UW enabled, include its parameters with no weight decay
        if self.uw_loss is not None:
            g_params = [
                {'params': self.generator.parameters()},
                {'params': self.uw_loss.parameters(), 'weight_decay': 0},
            ]
        else:
            g_params = self.generator.parameters()

        # Adam with pix2pix betas
        opt_g = Adam(
            g_params,
            lr=self.config.lr,
            betas=self.config.betas,
        )
        opt_d = Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
        )

        # CosineAnnealingLR for both
        # T_max is epochs after warmup
        t_max = self.config.max_epochs - self.config.warmup_epochs
        base_sched_g = CosineAnnealingLR(opt_g, T_max=max(t_max, 1))
        base_sched_d = CosineAnnealingLR(opt_d, T_max=max(t_max, 1))

        # Wrap with linear warmup
        sched_g = LinearWarmupScheduler(
            opt_g, base_sched_g, self.config.warmup_epochs
        )
        sched_d = LinearWarmupScheduler(
            opt_d, base_sched_d, self.config.warmup_epochs
        )

        return opt_g, opt_d, sched_g, sched_d

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device.

        Args:
            batch: Dictionary with tensor values.

        Returns:
            Batch with tensors on device.
        """
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def _init_wandb(self) -> None:
        """Initialize W&B run (rank 0 only).

        Creates a W&B run for experiment tracking if:
        - This is the main process (rank 0)
        - wandb is installed
        - WANDB_MODE is not 'disabled'
        """
        if not self._is_main_process():
            self.wandb_run = None
            return

        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed, skipping W&B initialization")
            self.wandb_run = None
            return

        # Check if wandb is disabled (for testing)
        if os.environ.get("WANDB_MODE") == "disabled":
            logger.info("W&B disabled via WANDB_MODE=disabled")
            self.wandb_run = None
            return

        # Prepare config dict
        config_dict = {}
        if OMEGACONF_AVAILABLE and hasattr(self.config, "__dataclass_fields__"):
            try:
                config_dict = OmegaConf.to_container(
                    OmegaConf.structured(self.config), resolve=True
                )
            except Exception:
                # Fallback to dict conversion
                config_dict = {
                    k: getattr(self.config, k)
                    for k in self.config.__dataclass_fields__
                }
        elif hasattr(self.config, "__dict__"):
            config_dict = vars(self.config)

        try:
            self.wandb_run = wandb.init(
                project=getattr(self.config, "wandb_project", "multiplex-virtual-ihc"),
                name=getattr(self.config, "experiment_name", "experiment"),
                config=config_dict,
                tags=["phase4", "training"],
                resume="allow",
            )
            logger.info(f"W&B initialized: {self.wandb_run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.wandb_run = None

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric names to values.
            step: Global training step.
        """
        if self.wandb_run is None:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")

    def _log_samples(
        self,
        bf: torch.Tensor,
        real: torch.Tensor,
        fake: torch.Tensor,
        step: int,
    ) -> None:
        """Log sample predictions as images to W&B.

        Args:
            bf: Input brightfield images [B, 1, H, W].
            real: Ground truth marker images [B, 5, H, W].
            fake: Predicted marker images [B, 5, H, W].
            step: Global training step.
        """
        if self.wandb_run is None:
            return

        num_samples = min(
            getattr(self.config, "num_samples", 4),
            bf.shape[0],
        )

        try:
            images = []
            for i in range(num_samples):
                # Normalize tensors to [0, 1] for visualization
                bf_img = bf[i, 0].detach().cpu().float()
                bf_img = (bf_img - bf_img.min()) / (bf_img.max() - bf_img.min() + 1e-8)

                # Average markers for visualization (or use first channel)
                real_avg = real[i].mean(dim=0).detach().cpu().float()
                real_avg = (real_avg - real_avg.min()) / (real_avg.max() - real_avg.min() + 1e-8)

                fake_avg = fake[i].mean(dim=0).detach().cpu().float()
                fake_avg = (fake_avg - fake_avg.min()) / (fake_avg.max() - fake_avg.min() + 1e-8)

                # Error map (absolute difference)
                error = torch.abs(real_avg - fake_avg)

                # Convert to numpy for wandb.Image
                images.append(
                    wandb.Image(
                        bf_img.numpy(),
                        caption=f"Sample {i}: Input BF",
                    )
                )
                images.append(
                    wandb.Image(
                        real_avg.numpy(),
                        caption=f"Sample {i}: Ground Truth (avg markers)",
                    )
                )
                images.append(
                    wandb.Image(
                        fake_avg.numpy(),
                        caption=f"Sample {i}: Predicted (avg markers)",
                    )
                )
                images.append(
                    wandb.Image(
                        error.numpy(),
                        caption=f"Sample {i}: Error Map",
                    )
                )

            wandb.log({"samples": images}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log samples to W&B: {e}")

    def _log_gpu_utilization(self, step: int) -> None:
        """Log GPU memory utilization to W&B.

        Args:
            step: Global training step.
        """
        if self.wandb_run is None:
            return

        if not torch.cuda.is_available():
            return

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)

            if max_allocated > 0:
                memory_pct = allocated / max_allocated * 100
                wandb.log({"gpu_memory_pct": memory_pct}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log GPU utilization: {e}")

    def _log_uw_weights(self, step: int) -> None:
        """Log uncertainty weights to W&B (epoch-level).

        Args:
            step: Global training step.
        """
        if self.uw_loss is None or self.wandb_run is None:
            return

        try:
            weight_dict = self.uw_loss.get_weight_dict()
            wandb.log(weight_dict, step=step)
        except Exception as e:
            logger.warning(f"Failed to log UW weights: {e}")

    def _finish_wandb(self) -> None:
        """Finish W&B run and cleanup."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
            self.wandb_run = None

    def _compute_loss_components(
        self,
        markers_fake: torch.Tensor,
        markers_real: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute individual loss components for gradient monitoring.

        Returns dict with 'recon' (L1 + perceptual) and 'bio' (biological constraints) losses.
        Only called when gradient monitoring is enabled.

        Args:
            markers_fake: Generated markers, shape (B, num_markers, H, W).
            markers_real: Ground truth markers, shape (B, num_markers, H, W).

        Returns:
            Dict with 'recon' and 'bio' loss tensors.
        """
        # Reconstruction losses (L1 + perceptual)
        l1_loss = nn.functional.l1_loss(markers_fake, markers_real)
        fake_rgb = self.multi_loss._prepare_for_lpips(markers_fake)
        real_rgb = self.multi_loss._prepare_for_lpips(markers_real)
        perc_loss = self.multi_loss.criterion_perc(fake_rgb, real_rgb).mean()
        recon_loss = self.config.lambda_l1 * l1_loss + self.config.lambda_perc * perc_loss

        # Biological constraint loss (if enabled and lambda_bio > 0)
        bio_loss = torch.tensor(0.0, device=markers_fake.device)
        if hasattr(self.multi_loss, 'bio_loss') and self.multi_loss.bio_loss is not None:
            if self.multi_loss.lambda_bio > 0:
                bio_result, _ = self.multi_loss.bio_loss(markers_fake, markers_real)
                bio_loss = self.multi_loss.lambda_bio * bio_result

        return {'recon': recon_loss, 'bio': bio_loss}

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step with D and G updates.

        Performs alternating discriminator and generator updates following
        the pix2pix training scheme.

        Args:
            batch: Dictionary with 'bf' and 'markers' tensors.

        Returns:
            Dictionary with all loss values.
        """
        self.generator.train()
        self.discriminator.train()

        batch = self._to_device(batch)
        bf = batch["bf"]
        markers_real = batch["markers"]

        metrics = {}

        # === Discriminator Step ===
        self.opt_d.zero_grad()

        with autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_bf16):
            # Generate fake markers
            markers_fake = self.generator(bf)

            # Discriminator predictions
            pred_real = self.discriminator(bf, markers_real)
            pred_fake = self.discriminator(bf, markers_fake.detach())  # Detach to not update G

            # Discriminator loss
            d_loss, d_loss_dict = self.disc_loss(pred_real, pred_fake)

        # Backward and step (no GradScaler needed for BF16)
        d_loss.backward()
        self.opt_d.step()

        metrics.update(d_loss_dict)

        # === Generator Step ===
        self.opt_g.zero_grad()

        with autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_bf16):
            # Re-compute discriminator prediction for fake (no detach)
            pred_fake_for_g = self.discriminator(bf, markers_fake)

            # Gradient conflict monitoring (if enabled)
            grad_metrics = {}
            if self.gradient_monitor is not None and self.multi_loss.lambda_bio > 0:
                # Compute individual loss components
                loss_components = self._compute_loss_components(
                    markers_fake, markers_real
                )
                # Compute gradient conflicts (only returns metrics every N steps)
                grad_metrics = self.gradient_monitor.compute_conflicts(
                    loss_components, self.generator
                ) or {}

            # Generator loss (L1 + perceptual + adversarial + bio)
            g_loss, g_loss_dict = self.multi_loss(markers_fake, markers_real, pred_fake_for_g)

        # Backward and step
        g_loss.backward()
        self.opt_g.step()

        metrics.update(g_loss_dict)
        if grad_metrics:
            metrics.update(grad_metrics)
            if self._is_main_process():
                self._log_metrics(grad_metrics, self.global_step)

        self.global_step += 1

        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one validation step (no gradient updates).

        Args:
            batch: Dictionary with 'bf' and 'markers' tensors.

        Returns:
            Dictionary with L1 and perceptual loss values.
        """
        self.generator.eval()

        batch = self._to_device(batch)
        bf = batch["bf"]
        markers_real = batch["markers"]

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_bf16):
                # Generate fake markers
                markers_fake = self.generator(bf)

                # Compute L1 loss
                l1_loss = nn.functional.l1_loss(markers_fake, markers_real)

                # Compute perceptual loss (uses internal LPIPS)
                fake_rgb = self.multi_loss._prepare_for_lpips(markers_fake)
                real_rgb = self.multi_loss._prepare_for_lpips(markers_real)
                perc_loss = self.multi_loss.criterion_perc(fake_rgb, real_rgb).mean()

        return {
            "val_l1": l1_loss.item(),
            "val_perc": perc_loss.item(),
            "val_loss": l1_loss.item() + perc_loss.item(),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with epoch-averaged metrics.
        """
        # Set sampler epoch for proper shuffling in DDP
        if hasattr(self.train_loader, "sampler") and isinstance(
            self.train_loader.sampler, DistributedSampler
        ):
            self.train_loader.sampler.set_epoch(epoch)

        # Check for uncertainty weight unfreezing
        if self.uw_loss is not None:
            if self.uw_loss.maybe_unfreeze(epoch):
                if self._is_main_process():
                    logger.info(f"Epoch {epoch}: Unfroze uncertainty weights")

        # Update lambda_bio based on curriculum schedule
        if self.curriculum_scheduler is not None:
            current_lambda_bio = self.curriculum_scheduler.get_lambda_bio(epoch)
            self.multi_loss.lambda_bio = current_lambda_bio

            # Log curriculum state (main process only)
            if self._is_main_process():
                curriculum_state = self.curriculum_scheduler.get_schedule_state(epoch)
                self._log_metrics(curriculum_state, self.global_step)

        self.generator.train()
        self.discriminator.train()

        epoch_metrics = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            step_metrics = self.training_step(batch)

            # Accumulate metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            num_batches += 1

            # Log every N steps
            if self._is_main_process() and (batch_idx + 1) % self.config.log_every_n_steps == 0:
                avg_loss = epoch_metrics.get("loss_total", 0.0) / num_batches
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss: {avg_loss:.4f}"
                )

                # Log metrics to W&B
                log_metrics = {
                    "train/loss_total": step_metrics.get("loss_total", 0),
                    "train/loss_g": step_metrics.get("loss_g", 0),
                    "train/loss_d": step_metrics.get("loss_d", 0),
                    "train/loss_l1": step_metrics.get("loss_l1", 0),
                    "train/loss_perc": step_metrics.get("loss_perc", 0),
                    "train/loss_gan": step_metrics.get("loss_gan", 0),
                    "train/lr_g": self.sched_g.get_last_lr()[0],
                    "train/lr_d": self.sched_d.get_last_lr()[0],
                    "train/epoch": epoch,
                }
                self._log_metrics(log_metrics, self.global_step)
                self._log_gpu_utilization(self.global_step)

            # Log sample images every N steps
            sample_every = getattr(self.config, "sample_every_n_steps", 500)
            if self._is_main_process() and self.global_step % sample_every == 0 and self.global_step > 0:
                batch_device = self._to_device(batch)
                bf = batch_device["bf"]
                markers_real = batch_device["markers"]
                with torch.no_grad():
                    markers_fake = self.generator(bf)
                self._log_samples(bf, markers_real, markers_fake, self.global_step)

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        # Log epoch summary to W&B
        if self._is_main_process():
            epoch_log = {
                f"epoch/{k}": v for k, v in epoch_metrics.items()
            }
            epoch_log["epoch/epoch"] = epoch
            epoch_log["epoch/lr_g"] = self.sched_g.get_last_lr()[0]
            epoch_log["epoch/lr_d"] = self.sched_d.get_last_lr()[0]
            self._log_metrics(epoch_log, self.global_step)

            # Log uncertainty weights (epoch-level per CONTEXT.md)
            if getattr(self.config, 'uw_log_every_epoch', True):
                self._log_uw_weights(self.global_step)

        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with validation metrics.
        """
        self.generator.eval()

        val_metrics = {}
        num_batches = 0

        for batch in self.val_loader:
            step_metrics = self.validation_step(batch)

            # Accumulate metrics
            for key, value in step_metrics.items():
                val_metrics[key] = val_metrics.get(key, 0.0) + value
            num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)

        if self._is_main_process():
            logger.info(
                f"Epoch {epoch} validation: "
                f"val_l1={val_metrics.get('val_l1', 0):.4f}, "
                f"val_perc={val_metrics.get('val_perc', 0):.4f}"
            )

            # Log validation metrics to W&B with val/ prefix
            val_log = {f"val/{k}": v for k, v in val_metrics.items()}
            val_log["val/epoch"] = epoch
            self._log_metrics(val_log, self.global_step)

        return val_metrics

    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Resume training from a specific checkpoint.

        Restores all training state including models, optimizers, schedulers,
        epoch, and global step. Sets start_epoch to epoch + 1 for continuation.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            True if resume successful, False otherwise.
        """
        if not os.path.exists(checkpoint_path):
            if self._is_main_process():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        if self._is_main_process():
            logger.info(f"Resuming from: {checkpoint_path}")

        epoch, step, metrics = self.checkpoint_manager.load(
            path=checkpoint_path,
            generator=self.generator,
            discriminator=self.discriminator,
            opt_g=self.opt_g,
            opt_d=self.opt_d,
            sched_g=self.sched_g,
            sched_d=self.sched_d,
            device=self.device,
        )

        self.start_epoch = epoch + 1  # Resume from next epoch
        self.global_step = step

        if self._is_main_process():
            logger.info(f"Resumed from epoch {epoch}, step {step}")
            logger.info(f"Will start training from epoch {self.start_epoch}")
            if metrics:
                logger.info(f"Last metrics: {metrics}")

        return True

    def resume_from_latest(self) -> bool:
        """Resume from the most recent checkpoint if available.

        Finds the latest checkpoint in the checkpoint directory and
        resumes training from it.

        Returns:
            True if resumed, False if no checkpoint found.
        """
        latest = self.checkpoint_manager.get_latest()
        if latest is None:
            if self._is_main_process():
                logger.info("No checkpoint found, starting fresh")
            return False
        return self.resume_from_checkpoint(latest)

    def fit(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        """Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from. If provided,
                calls resume_from_checkpoint() before starting training.
                Alternatively, call resume_from_checkpoint() or
                resume_from_latest() before fit() for more control.

        Returns:
            Dictionary with final training metrics.
        """
        # Resume from checkpoint if specified (backward compatibility)
        if resume_from:
            self.resume_from_checkpoint(resume_from)

        # Use self.start_epoch (set by resume methods or default 0)
        final_metrics = {}

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate(epoch)

            # Update schedulers
            self.sched_g.step()
            self.sched_d.step()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            final_metrics = all_metrics

            # Checkpoint every N epochs
            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                self.checkpoint_manager.save(
                    self.generator,
                    self.discriminator,
                    self.opt_g,
                    self.opt_d,
                    self.sched_g,
                    self.sched_d,
                    epoch,
                    self.global_step,
                    all_metrics,
                    self.config,
                )

            # Save best model
            self.checkpoint_manager.save_best(
                self.generator,
                self.discriminator,
                self.opt_g,
                self.opt_d,
                self.sched_g,
                self.sched_d,
                epoch,
                self.global_step,
                val_metrics,
                self.config,
                metric_key="val_loss",
                mode="min",
            )

            if self._is_main_process():
                lr = self.sched_g.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} complete: "
                    f"train_loss={train_metrics.get('loss_total', 0):.4f}, "
                    f"val_loss={val_metrics.get('val_loss', 0):.4f}, "
                    f"lr={lr:.6f}"
                )

        # Final checkpoint
        self.checkpoint_manager.save(
            self.generator,
            self.discriminator,
            self.opt_g,
            self.opt_d,
            self.sched_g,
            self.sched_d,
            self.config.max_epochs - 1,
            self.global_step,
            final_metrics,
            self.config,
        )

        # Finish W&B run
        self._finish_wandb()

        logger.info("Training complete")
        return final_metrics
