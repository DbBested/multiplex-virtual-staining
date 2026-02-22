"""JiT training loop for flow matching.

Composes FlowMatchingLoss, EMA, and ODE samplers into a complete training
pipeline for ConditionalJiT. This is purpose-built for the flow matching
paradigm (single model, regression loss, EMA) -- not adapted from the V3
GAN trainer.

Training recipe follows DiT/SiT:
- AdamW with lr=1e-4, betas=(0.9, 0.999), weight_decay=0, grad_clip=1.0
- EMA decay 0.9999
- Step-based training with infinite dataloader cycling
- Logit-normal timestep sampling (SD3)
- Velocity prediction (SiT/SD3 standard)

Reference:
    DiT: Scalable Diffusion Models with Transformers
    SiT: Exploring Flow and Diffusion-based Generative Models
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision.utils import make_grid

import torch.nn.functional as F

from multiplex.models.discriminator import PatchGAN70
from multiplex.models.multi_stain_encoder import ModalityDropout
from multiplex.training.ema import EMA
from multiplex.training.flow_matching import (
    FlowMatchingLoss,
    bridge_sample,
    direct_predict,
    euler_sample,
    heun_sample,
)

logger = logging.getLogger(__name__)

# Channel names for per-channel evaluation metrics (COND-04)
CHANNEL_NAMES = ['DAPI', 'Lap2', 'Marker']

# Optional imports -- graceful fallback if unavailable
try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )

    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

try:
    import pyiqa

    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False


class JiTTrainer:
    """Training loop for ConditionalJiT with flow matching.

    Manages the full training lifecycle: forward pass, loss computation,
    gradient clipping, optimizer step, EMA update, evaluation with EMA
    weights, sample generation, checkpointing, and W&B logging.

    Args:
        model: ConditionalJiT model (already on device).
        config: OmegaConf / namespace with training hyperparameters.
            Required keys: lr, betas, weight_decay, grad_clip, prediction,
            ema_decay, use_bf16, log_every_n_steps, eval_every_n_steps,
            sample_every_n_steps, checkpoint_every_n_steps, checkpoint_dir,
            euler_steps, num_samples.
        device: torch.device to use.
        wandb_run: Active W&B run object, or None to disable logging.

    Example:
        >>> model = ConditionalJiT(...).to(device)
        >>> trainer = JiTTrainer(model, cfg, device)
        >>> trainer.train(train_loader, val_loader, max_steps=50000)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        wandb_run: Any = None,
        rank: int = 0,
        world_size: int = 1,
        unwrapped_model: nn.Module | None = None,
        train_sampler: Any = None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.wandb_run = wandb_run
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler

        # Per-channel names from config (configurable via target_channels).
        # Falls back to module-level CHANNEL_NAMES for backward compatibility
        # with old checkpoints that don't store target_channels in config.
        cfg_channels = getattr(config, "target_channels", None)
        if cfg_channels is not None:
            self.channel_names = list(cfg_channels)
        else:
            self.channel_names = list(CHANNEL_NAMES)

        # EMA and checkpointing use the unwrapped model to avoid DDP's
        # module. prefix in parameter names. DDP params share storage
        # with unwrapped params, so the optimizer can use either.
        self._ema_model = unwrapped_model if unwrapped_model is not None else model

        # Optimizer -- only parameters that require grad (encoder is frozen)
        # DDP params share storage with unwrapped model, either works
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=tuple(config.betas),
            weight_decay=config.weight_decay,
        )

        # Flow matching loss (with optional bridge matching)
        use_bridge = getattr(config, 'use_bridge', False)
        bridge_sigma = getattr(config, 'bridge_sigma', 0.1)
        self.loss_fn = FlowMatchingLoss(
            prediction_type=config.prediction,
            bridge=use_bridge,
            bridge_sigma=bridge_sigma,
            loss_type=getattr(config, 'loss_type', 'mse'),
        )

        # EMA -- MUST use unwrapped model for consistent parameter names
        self.ema = EMA(self._ema_model, decay=config.ema_decay)

        # State
        self.step = 0
        self.best_psnr = 0.0

        # Modality dropout for variable input configurations (COND-03)
        # NOT a submodule of self.model -- instantiated on JiTTrainer directly
        # so model.train()/model.eval() does not affect it.
        self.modality_dropout = ModalityDropout(
            p_ihc_only=getattr(config, 'p_ihc_only', 0.3),
            p_ihc_h=getattr(config, 'p_ihc_h', 0.4),
        )

        # BioLoss (Phase 24: conditional, only when lambda_bio > 0)
        self.bio_loss_fn = None
        lambda_bio = getattr(config, 'lambda_bio', 0.0)
        if lambda_bio > 0:
            from multiplex.training.bio_losses import BioLossSuite
            self.bio_loss_fn = BioLossSuite(
                weight_nuclear=getattr(config, 'bio_weight_nuclear', 1.0),
                weight_coherence=getattr(config, 'bio_weight_coherence', 1.0),
                enable_nuclear=getattr(config, 'bio_enable_nuclear', True),
                enable_coherence=getattr(config, 'bio_enable_coherence', True),
                t_threshold=getattr(config, 'bio_t_threshold', 0.3),
                use_gt_dapi=getattr(config, 'bio_use_gt_dapi', True),
                dilation_kernel_size=getattr(config, 'bio_dilation_kernel', 7),
                nuclear_threshold=getattr(config, 'bio_nuclear_threshold', 0.3),
                nuclear_temperature=getattr(config, 'bio_nuclear_temperature', 10.0),
                use_distance_weighting=getattr(config, 'bio_use_distance_weighting', True),
                margin_pixels=getattr(config, 'bio_margin_pixels', 10),
            ).to(device)
            logger.info(
                f"BioLoss enabled: lambda={lambda_bio}, "
                f"nuclear={getattr(config, 'bio_enable_nuclear', True)}, "
                f"coherence={getattr(config, 'bio_enable_coherence', True)}"
            )

        # LPIPS training loss (PixelGen recipe -- no timestep gating)
        self.lpips_fn = None
        lambda_lpips = getattr(config, 'lambda_lpips', 0.0)
        if lambda_lpips > 0:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg', spatial=False).to(device)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            logger.info(f"LPIPS training loss enabled: lambda={lambda_lpips}")

        # P-DINO perceptual loss (PixelGen recipe -- DINOv2 patch features)
        self.pdino_fn = None
        lambda_pdino = getattr(config, 'lambda_pdino', 0.0)
        if lambda_pdino > 0:
            from multiplex.training.perceptual import PDINOLoss
            self.pdino_fn = PDINOLoss(device=device).to(device)
            logger.info(f"P-DINO training loss enabled: lambda={lambda_pdino}")

        # Adversarial training (optional PatchGAN70 discriminator)
        self.disc = None
        self.disc_optimizer = None
        self._disc_unwrapped = None
        self.lambda_adv = getattr(config, 'lambda_adv', 0.0)

        if self.lambda_adv > 0:
            disc_in_ch = 4 + getattr(config, 'out_chans', 3)  # source(4) + target(3)
            self._disc_unwrapped = PatchGAN70(
                in_channels=disc_in_ch,
                ndf=64,
                use_spectral_norm=True,
            ).to(device)

            # Don't DDP-wrap the discriminator — having two DDP modules
            # causes deadlocks at barriers when D is inactive during warmup.
            # Each rank trains its own D replica independently. Since all
            # ranks see similar data distributions, the D replicas converge
            # to similar solutions and provide valid gradient signals to
            # the DDP-synced generator.
            self.disc = self._disc_unwrapped

            self.disc_optimizer = torch.optim.Adam(
                self._disc_unwrapped.parameters(),
                lr=getattr(config, 'disc_lr', 2e-4),
                betas=(0.0, 0.9),
            )

            self.adv_warmup_steps = getattr(config, 'adv_warmup_steps', 5000)
            self.adv_t_threshold = getattr(config, 'adv_t_threshold', 0.5)
            self.r1_gamma = getattr(config, 'r1_gamma', 0.05)
            self.r1_interval = getattr(config, 'r1_interval', 16)

            disc_params = sum(p.numel() for p in self._disc_unwrapped.parameters())
            logger.info(
                f"Adversarial training enabled: lambda_adv={self.lambda_adv}, "
                f"warmup={self.adv_warmup_steps} steps, "
                f"D params={disc_params / 1e6:.1f}M"
            )

        # Metrics (lazy init on first evaluate call)
        self._psnr_metric = None
        self._ssim_metric = None
        self._lpips_metric = None

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    @staticmethod
    def _set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
        """Toggle requires_grad for all parameters of a model."""
        for p in model.parameters():
            p.requires_grad = requires_grad

    def train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        avail_config: torch.Tensor,
        hematoxylin: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Execute a single training step.

        Forward pass, loss, backward, gradient clipping, optimizer step,
        EMA update. When adversarial training is enabled, also performs
        a discriminator update step before the generator step.

        Args:
            source: Source image (B, C_src, H, W). Typically 4-channel
                (IHC RGB + Hematoxylin) after ModalityDropout masking.
            target: Clean target image (B, C_out, H, W).
            avail_config: Availability config (B,) from ModalityDropout.
            hematoxylin: Original hematoxylin channel (B, 1, H, W) for
                BIO-02 spatial coherence loss. None when unavailable.

        Returns:
            Dict with loss, loss_fm, grad_norm, step, and bio/adv metrics.
        """
        self.model.train()

        # Forward with optional mixed precision
        if getattr(self.config, "use_bf16", False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = self.loss_fn(self.model, source, target, avail_config)
        else:
            result = self.loss_fn(self.model, source, target, avail_config)

        loss_fm = result["loss"]
        x1_hat = result["x1_hat"]
        timesteps = result["timesteps"]

        adv_active = (
            self.disc is not None
            and self.step >= self.adv_warmup_steps
        )

        # ==============================================================
        # Discriminator step (D update with detached generator output)
        # ==============================================================
        adv_metrics = {}
        if adv_active:
            self.disc.train()
            self._set_requires_grad(self.disc, True)

            x1_hat_d = x1_hat.detach()

            # D on real: source || target
            d_real = self.disc(source.detach(), target.detach())
            # D on fake: source || x1_hat (detached)
            d_fake = self.disc(source.detach(), x1_hat_d)

            # Hinge loss
            d_loss_real = F.relu(1.0 - d_real).mean()
            d_loss_fake = F.relu(1.0 + d_fake).mean()
            d_loss = (d_loss_real + d_loss_fake) / 2

            # R1 gradient penalty (lazy regularization)
            r1_val = 0.0
            if self.step % self.r1_interval == 0:
                target_r1 = target.detach().requires_grad_(True)
                d_real_r1 = self.disc(source.detach(), target_r1)
                r1_grads = torch.autograd.grad(
                    outputs=d_real_r1.sum(),
                    inputs=target_r1,
                    create_graph=True,
                )[0]
                r1_penalty = r1_grads.pow(2).reshape(
                    r1_grads.shape[0], -1
                ).sum(1).mean()
                d_loss = d_loss + (self.r1_gamma / 2) * r1_penalty * self.r1_interval
                r1_val = r1_penalty.item()

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            adv_metrics = {
                "loss_d": d_loss.item(),
                "loss_d_real": d_loss_real.item(),
                "loss_d_fake": d_loss_fake.item(),
                "d_r1": r1_val,
            }

        # ==============================================================
        # Generator step (flow matching + auxiliary + adversarial G loss)
        # ==============================================================
        loss = loss_fm

        # Bio loss (Phase 24): auxiliary losses on predicted clean image
        bio_metrics = {}
        if self.bio_loss_fn is not None:
            x1_hat_f32 = x1_hat.float().clamp(0, 1)
            target_f32 = target.float()
            h_f32 = hematoxylin.float() if hematoxylin is not None else None

            bio_loss, bio_dict = self.bio_loss_fn(x1_hat_f32, target_f32, h_f32, timesteps)
            loss = loss + getattr(self.config, 'lambda_bio', 0.05) * bio_loss
            bio_metrics = bio_dict

        # Perceptual losses on predicted clean image (PixelGen recipe).
        # With x-prediction, x1_hat is always a clean image estimate at any t,
        # so perceptual losses apply at ALL timesteps (no gating).
        if self.lpips_fn is not None:
            x1_hat_f32 = x1_hat.float().clamp(0, 1) * 2 - 1
            target_f32 = target.float().clamp(0, 1) * 2 - 1
            lpips_val = self.lpips_fn(x1_hat_f32, target_f32).mean()
            loss = loss + getattr(self.config, 'lambda_lpips', 0.01) * lpips_val
            bio_metrics["loss_lpips"] = lpips_val.item()

        if self.pdino_fn is not None:
            pdino_val = self.pdino_fn(x1_hat.float().clamp(0, 1), target.float())
            loss = loss + getattr(self.config, 'lambda_pdino', 0.01) * pdino_val
            bio_metrics["loss_pdino"] = pdino_val.item()

        # Adversarial G loss: fool the discriminator
        if adv_active:
            self._set_requires_grad(self.disc, False)

            # Timestep gating: only apply when t > threshold
            adv_mask = (timesteps > self.adv_t_threshold).float()
            gate_frac = adv_mask.sum() / adv_mask.numel()

            if gate_frac > 0:
                d_fake_g = self.disc(source, x1_hat)
                # Hinge G loss: maximize D output → minimize -D(fake)
                g_adv_loss = -d_fake_g.mean()
                loss = loss + self.lambda_adv * gate_frac * g_adv_loss
                adv_metrics["loss_g_adv"] = g_adv_loss.item()
                adv_metrics["adv_gate_frac"] = gate_frac.item()

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_clip = getattr(self.config, "grad_clip", 0.0)
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            ).item()
        else:
            grad_norm = 0.0

        # Optimizer step
        self.optimizer.step()

        # EMA update (uses unwrapped model for consistent param names)
        self.ema.update(self._ema_model)

        self.step += 1

        metrics = {
            "loss": loss.item(),
            "loss_fm": loss_fm.item(),
            "grad_norm": grad_norm,
            "step": self.step,
        }
        metrics.update(bio_metrics)
        metrics.update(adv_metrics)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        """Lazily initialize torchmetrics on the correct device."""
        if HAS_TORCHMETRICS and self._psnr_metric is None:
            self._psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(
                self.device
            )
            self._ssim_metric = StructuralSimilarityIndexMeasure(
                data_range=1.0
            ).to(self.device)
        if HAS_PYIQA and self._lpips_metric is None:
            self._lpips_metric = pyiqa.create_metric(
                "lpips-vgg", device=self.device
            )

    @staticmethod
    def _pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Pearson correlation coefficient between pred and target.

        Args:
            pred: Predicted tensor (B, C, H, W) in [0, 1].
            target: Target tensor (B, C, H, W) in [0, 1].

        Returns:
            Mean PCC across the batch.
        """
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        num = (pred_centered * target_centered).sum(dim=1)
        denom = (pred_centered.norm(dim=1) * target_centered.norm(dim=1)).clamp(min=1e-8)
        return (num / denom).mean().item()

    def evaluate(
        self,
        val_dataloader: torch.utils.data.DataLoader,
        num_steps: int = 50,
    ) -> dict[str, float]:
        """Evaluate model quality using EMA weights.

        Generates samples with Euler ODE sampler and computes aggregate and
        per-channel PSNR/SSIM/LPIPS/PCC. Evaluates in IHC+H mode (avail=1)
        as the primary evaluation configuration.

        Args:
            val_dataloader: Validation data loader.
            num_steps: Number of Euler integration steps.

        Returns:
            Dict with aggregate psnr, ssim, lpips, pcc and per-channel metrics.
        """
        self._init_metrics()
        self.ema.apply_shadow(self._ema_model)
        self._ema_model.eval()

        try:
            psnr_sum = 0.0
            ssim_sum = 0.0
            lpips_sum = 0.0
            pcc_sum = 0.0
            count = 0

            # Per-channel accumulators
            per_ch_psnr = {name: 0.0 for name in self.channel_names}
            per_ch_ssim = {name: 0.0 for name in self.channel_names}
            per_ch_lpips = {name: 0.0 for name in self.channel_names}
            per_ch_pcc = {name: 0.0 for name in self.channel_names}

            for batch in val_dataloader:
                # Construct 4-channel source: IHC RGB + Hematoxylin
                source_4ch = torch.cat([
                    batch["ihc"].to(self.device),
                    batch["hematoxylin"].to(self.device),
                ], dim=1)
                target = batch["targets"].to(self.device)
                B = source_4ch.shape[0]

                # Evaluate in IHC+H mode (avail=1)
                avail = torch.ones(
                    B, dtype=torch.long, device=self.device
                )

                use_bridge = getattr(self.config, 'use_bridge', False)
                prediction_type = getattr(self.config, "prediction", "velocity")
                sampler_cfg = getattr(self.config, 'sampler', 'auto')

                # Determine whether to use direct prediction
                use_direct = (
                    sampler_cfg == 'direct'
                    or (sampler_cfg == 'auto' and prediction_type == 'x_prediction')
                )

                if use_bridge and not use_direct:
                    pred = bridge_sample(
                        self._ema_model,
                        source_4ch,
                        avail,
                        num_steps=num_steps,
                        prediction_type=prediction_type,
                    )
                elif use_direct:
                    pred = direct_predict(
                        self._ema_model,
                        source_4ch,
                        avail,
                        num_steps=num_steps,
                        img_size=getattr(self.config, "img_size", 512),
                        out_chans=getattr(self.config, "out_chans", 3),
                        prediction_type=prediction_type,
                    )
                else:
                    pred = euler_sample(
                        self._ema_model,
                        source_4ch,
                        avail,
                        num_steps=num_steps,
                        img_size=getattr(self.config, "img_size", 512),
                        out_chans=getattr(self.config, "out_chans", 3),
                        prediction_type=prediction_type,
                    )

                # Aggregate metrics
                if HAS_TORCHMETRICS:
                    psnr_sum += self._psnr_metric(pred, target).item()
                    ssim_sum += self._ssim_metric(pred, target).item()
                else:
                    mse = torch.nn.functional.mse_loss(pred, target)
                    psnr_sum += (
                        -10.0 * torch.log10(mse.clamp(min=1e-10))
                    ).item()
                    ssim_sum += 0.0

                # Aggregate PCC
                pcc_sum += self._pearson_corr(pred, target)

                # Per-channel PSNR/SSIM/PCC
                for c, name in enumerate(self.channel_names):
                    pred_ch = pred[:, c:c+1]
                    target_ch = target[:, c:c+1]
                    per_ch_pcc[name] += self._pearson_corr(pred_ch, target_ch)
                    if HAS_TORCHMETRICS:
                        per_ch_psnr[name] += self._psnr_metric(
                            pred_ch, target_ch
                        ).item()
                        per_ch_ssim[name] += self._ssim_metric(
                            pred_ch, target_ch
                        ).item()

                # Aggregate LPIPS (3-channel output)
                if self._lpips_metric is not None:
                    p_lpips = pred.clamp(0, 1)
                    t_lpips = target.clamp(0, 1)
                    if p_lpips.shape[1] == 3:
                        lpips_sum += self._lpips_metric(
                            p_lpips, t_lpips
                        ).mean().item()
                    else:
                        # Per-channel LPIPS averaged for non-3ch outputs
                        ch_lpips_vals = []
                        for c in range(p_lpips.shape[1]):
                            pc = p_lpips[:, c:c+1].expand(-1, 3, -1, -1)
                            tc = t_lpips[:, c:c+1].expand(-1, 3, -1, -1)
                            ch_lpips_vals.append(
                                self._lpips_metric(pc, tc).mean().item()
                            )
                        lpips_sum += sum(ch_lpips_vals) / len(ch_lpips_vals)

                    # Per-channel LPIPS
                    for c, name in enumerate(self.channel_names):
                        pc = p_lpips[:, c:c+1].expand(-1, 3, -1, -1)
                        tc = t_lpips[:, c:c+1].expand(-1, 3, -1, -1)
                        per_ch_lpips[name] += self._lpips_metric(
                            pc, tc
                        ).mean().item()

                count += 1

            n = max(count, 1)
            metrics = {
                "psnr": psnr_sum / n,
                "ssim": ssim_sum / n,
                "lpips": lpips_sum / n,
                "pcc": pcc_sum / n,
            }
            # Add per-channel metrics
            for name in self.channel_names:
                metrics[f"psnr_{name}"] = per_ch_psnr[name] / n
                metrics[f"ssim_{name}"] = per_ch_ssim[name] / n
                metrics[f"lpips_{name}"] = per_ch_lpips[name] / n
                metrics[f"pcc_{name}"] = per_ch_pcc[name] / n

        finally:
            self.ema.restore(self._ema_model)
            self._ema_model.train()

        return metrics

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples(
        self,
        source: torch.Tensor,
        avail_config: torch.Tensor,
        num_steps: int = 50,
        use_heun: bool = False,
    ) -> torch.Tensor:
        """Generate samples using EMA weights.

        Args:
            source: Source image (B, C_src, H, W). Typically 4-channel
                (IHC RGB + Hematoxylin) for full conditioning.
            avail_config: Availability config (B,).
            num_steps: Number of ODE integration steps.
            use_heun: If True, use Heun sampler; otherwise Euler.

        Returns:
            Generated images clamped to [0, 1].
        """
        self.ema.apply_shadow(self._ema_model)
        self._ema_model.eval()

        try:
            use_bridge = getattr(self.config, 'use_bridge', False)
            prediction_type = getattr(self.config, "prediction", "velocity")
            sampler_cfg = getattr(self.config, 'sampler', 'auto')

            # Determine whether to use direct prediction
            use_direct = (
                sampler_cfg == 'direct'
                or (sampler_cfg == 'auto' and prediction_type == 'x_prediction')
            )

            if use_bridge and not use_direct:
                pred = bridge_sample(
                    self._ema_model,
                    source,
                    avail_config,
                    num_steps=num_steps,
                    prediction_type=prediction_type,
                )
            elif use_direct:
                pred = direct_predict(
                    self._ema_model,
                    source,
                    avail_config,
                    num_steps=num_steps,
                    img_size=getattr(self.config, "img_size", 512),
                    out_chans=getattr(self.config, "out_chans", 3),
                    prediction_type=prediction_type,
                )
            else:
                sampler = heun_sample if use_heun else euler_sample
                pred = sampler(
                    self._ema_model,
                    source,
                    avail_config,
                    num_steps=num_steps,
                    img_size=getattr(self.config, "img_size", 512),
                    out_chans=getattr(self.config, "out_chans", 3),
                    prediction_type=prediction_type,
                )
        finally:
            self.ema.restore(self._ema_model)
            self._ema_model.train()

        return pred.clamp(0, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        metrics: dict[str, float] | None = None,
        lightweight: bool = False,
    ) -> None:
        """Save training state to disk.

        Saves model, EMA, step counter, config, and metrics. Optionally
        includes optimizer state for training resumption (``lightweight=False``).
        Best-model checkpoints are always lightweight (no optimizer) to save
        disk space -- they are only used for inference.

        Args:
            path: File path for the checkpoint.
            metrics: Optional validation metrics to store.
            lightweight: If True, skip optimizer/discriminator state to reduce
                file size (~40% smaller). Best-model checkpoints always use
                lightweight format regardless of this flag.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = (
            OmegaConf.to_container(self.config, resolve=True)
            if hasattr(self.config, "_metadata")  # OmegaConf DictConfig
            else (
                {k: v for k, v in vars(self.config).items() if not k.startswith("_")}
                if hasattr(self.config, "__dict__")
                else {}
            )
        )

        checkpoint = {
            "model_state_dict": self._ema_model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "step": self.step,
            "config": config_dict,
            "metrics": metrics or {},
            "best_psnr": self.best_psnr,
        }

        if not lightweight:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            # Save discriminator state if adversarial training is active
            if self._disc_unwrapped is not None:
                checkpoint["disc_state_dict"] = self._disc_unwrapped.state_dict()
                checkpoint["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path} (step {self.step})")

        # Save best checkpoint (lightweight -- inference only, no optimizer)
        if metrics and metrics.get("psnr", 0) > self.best_psnr:
            self.best_psnr = metrics["psnr"]
            best_path = path.parent / "best_model.pt"
            best_checkpoint = {
                "model_state_dict": checkpoint["model_state_dict"],
                "ema_state_dict": checkpoint["ema_state_dict"],
                "step": self.step,
                "config": config_dict,
                "metrics": metrics,
                "best_psnr": self.best_psnr,
            }
            torch.save(best_checkpoint, best_path)
            logger.info(
                f"New best model: PSNR={self.best_psnr:.2f} -> {best_path}"
            )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load training state from disk.

        Restores model, EMA, optimizer, step counter.

        Args:
            path: Path to checkpoint file.

        Returns:
            Step number from checkpoint (for resume).
        """
        checkpoint = torch.load(path, map_location=self.device)

        self._ema_model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        self.step = checkpoint["step"]
        self.best_psnr = checkpoint.get("best_psnr", 0.0)

        # Restore optimizer state (may be absent in lightweight checkpoints)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("No optimizer state in checkpoint (lightweight). "
                           "Optimizer will start fresh.")

        # Restore discriminator state if available
        if self._disc_unwrapped is not None and "disc_state_dict" in checkpoint:
            self._disc_unwrapped.load_state_dict(checkpoint["disc_state_dict"])
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
            logger.info("Restored discriminator state from checkpoint")

        logger.info(f"Loaded checkpoint: {path} (step {self.step})")
        return self.step

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        max_steps: int,
    ) -> None:
        """Run step-based training loop with DDP support.

        Iterates through train_loader with proper epoch tracking for
        DistributedSampler. Periodically logs metrics, runs evaluation,
        generates samples, and saves checkpoints (rank 0 only).

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            max_steps: Total number of training steps.
        """
        log_every = getattr(self.config, "log_every_n_steps", 50)
        eval_every = getattr(self.config, "eval_every_n_steps", 1000)
        sample_every = getattr(self.config, "sample_every_n_steps", 2000)
        ckpt_every = getattr(self.config, "checkpoint_every_n_steps", 5000)
        ckpt_dir = Path(getattr(self.config, "checkpoint_dir", "checkpoints/jit"))
        euler_steps = getattr(self.config, "euler_steps", 50)
        num_samples = getattr(self.config, "num_samples", 4)

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Epoch-tracked iteration (replaces cycle() for DistributedSampler)
        epoch_counter = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch_counter)
        data_iter = iter(train_loader)

        # Hold a fixed validation batch for sample generation (rank 0 only)
        if self.rank == 0:
            val_batch = next(iter(val_loader))
            sample_source_4ch = torch.cat([
                val_batch["ihc"][:num_samples],
                val_batch["hematoxylin"][:num_samples],
            ], dim=1).to(self.device)
            sample_target = val_batch["targets"][:num_samples].to(self.device)
            sample_avail = torch.ones(
                sample_source_4ch.shape[0], dtype=torch.long, device=self.device
            )

        start_step = self.step
        start_time = time.time()
        running_loss = 0.0

        if self.rank == 0:
            print(f"Starting training from step {start_step} to {max_steps}")
            print(f"  Log every {log_every} | Eval every {eval_every} | "
                  f"Sample every {sample_every} | Checkpoint every {ckpt_every}")

        try:
            while self.step < max_steps:
                # Get next batch, cycling with epoch tracking
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch_counter += 1
                    if self.train_sampler is not None:
                        self.train_sampler.set_epoch(epoch_counter)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                ihc = batch["ihc"].to(self.device)
                hema = batch["hematoxylin"].to(self.device)
                source_4ch = torch.cat([ihc, hema], dim=1)  # (B, 4, H, W)
                target = batch["targets"].to(self.device)

                # Apply modality dropout: randomly mask H and get avail config
                self.modality_dropout.train()
                masked_source, avail_config = self.modality_dropout(source_4ch)

                result = self.train_step(masked_source, target, avail_config, hematoxylin=hema)
                running_loss += result["loss"]

                # -- Periodic logging (rank 0 only) --
                if self.step % log_every == 0 and self.rank == 0:
                    avg_loss = running_loss / log_every
                    elapsed = time.time() - start_time
                    steps_per_sec = (self.step - start_step) / max(elapsed, 1)
                    lr = self.optimizer.param_groups[0]["lr"]

                    bio_str = f" | bio={result.get('loss_bio_total', 0):.4f}" if "loss_bio_total" in result else ""
                    lpips_str = f" | lpips={result.get('loss_lpips', 0):.4f}" if "loss_lpips" in result else ""
                    pdino_str = f" | pdino={result.get('loss_pdino', 0):.4f}" if "loss_pdino" in result else ""
                    adv_str = ""
                    if "loss_d" in result:
                        adv_str = (
                            f" | D={result['loss_d']:.4f}"
                            f" G_adv={result.get('loss_g_adv', 0):.4f}"
                        )
                    print(
                        f"Step {self.step}/{max_steps} | "
                        f"loss={avg_loss:.4f}{bio_str}{lpips_str}{pdino_str}{adv_str} | "
                        f"grad_norm={result['grad_norm']:.4f} | "
                        f"lr={lr:.2e} | "
                        f"{steps_per_sec:.1f} steps/s"
                    )

                    if self.wandb_run is not None:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/grad_norm": result["grad_norm"],
                            "train/lr": lr,
                            "train/step": self.step,
                        }
                        if "loss_bio_total" in result:
                            log_dict["train/loss_bio"] = result["loss_bio_total"]
                            log_dict["train/loss_fm"] = result.get("loss_fm", result["loss"])
                            if "loss_nuclear_consistency" in result:
                                log_dict["train/loss_nuclear"] = result["loss_nuclear_consistency"]
                            if "loss_spatial_coherence" in result:
                                log_dict["train/loss_coherence"] = result["loss_spatial_coherence"]
                            if "bio_gate_fraction" in result:
                                log_dict["train/bio_gate_frac"] = result["bio_gate_fraction"]
                        if "loss_lpips" in result:
                            log_dict["train/loss_lpips"] = result["loss_lpips"]
                        if "loss_pdino" in result:
                            log_dict["train/loss_pdino"] = result["loss_pdino"]
                        # Adversarial metrics
                        if "loss_d" in result:
                            log_dict["train/loss_d"] = result["loss_d"]
                            log_dict["train/loss_d_real"] = result["loss_d_real"]
                            log_dict["train/loss_d_fake"] = result["loss_d_fake"]
                        if "loss_g_adv" in result:
                            log_dict["train/loss_g_adv"] = result["loss_g_adv"]
                        if "adv_gate_frac" in result:
                            log_dict["train/adv_gate_frac"] = result["adv_gate_frac"]
                        if "d_r1" in result and result["d_r1"] > 0:
                            log_dict["train/d_r1"] = result["d_r1"]
                        self.wandb_run.log(log_dict, step=self.step)

                    running_loss = 0.0

                # -- Periodic evaluation (rank 0 only, barrier for DDP) --
                if self.step % eval_every == 0:
                    if self.rank == 0:
                        metrics = self.evaluate(val_loader, num_steps=euler_steps)
                        ch_psnr_str = " | ".join(
                            f"{name}={metrics.get(f'psnr_{name}', 0):.2f}"
                            for name in self.channel_names
                        )
                        ch_pcc_str = " | ".join(
                            f"{name}={metrics.get(f'pcc_{name}', 0):.4f}"
                            for name in self.channel_names
                        )
                        print(
                            f"  [eval] PSNR={metrics['psnr']:.2f} "
                            f"({ch_psnr_str}) | "
                            f"SSIM={metrics['ssim']:.4f} | "
                            f"PCC={metrics['pcc']:.4f} ({ch_pcc_str}) | "
                            f"LPIPS={metrics['lpips']:.4f}"
                        )

                        if self.wandb_run is not None:
                            log_dict = {
                                "val/psnr": metrics["psnr"],
                                "val/ssim": metrics["ssim"],
                                "val/lpips": metrics["lpips"],
                                "val/pcc": metrics["pcc"],
                            }
                            # Add per-channel metrics
                            for name in self.channel_names:
                                for metric_key in ['psnr', 'ssim', 'lpips', 'pcc']:
                                    full_key = f'{metric_key}_{name}'
                                    if full_key in metrics:
                                        log_dict[f'val/{full_key}'] = metrics[full_key]
                            self.wandb_run.log(log_dict, step=self.step)
                    if self.world_size > 1:
                        import torch.distributed as dist
                        dist.barrier()

                # -- Periodic sample generation (rank 0 only, barrier for DDP) --
                if self.step % sample_every == 0:
                    if self.rank == 0:
                        preds = self.generate_samples(
                            sample_source_4ch, sample_avail, num_steps=euler_steps
                        )
                        self._log_sample_images(
                            sample_source_4ch, preds, sample_target
                        )
                    if self.world_size > 1:
                        import torch.distributed as dist
                        dist.barrier()

                # -- Periodic checkpointing (rank 0 saves, all ranks barrier) --
                if self.step % ckpt_every == 0:
                    if self.rank == 0:
                        ckpt_path = ckpt_dir / f"step_{self.step:07d}.pt"
                        self.save_checkpoint(ckpt_path)
                        self._cleanup_checkpoints(ckpt_dir)
                    if self.world_size > 1:
                        import torch.distributed as dist
                        dist.barrier()

        except KeyboardInterrupt:
            if self.rank == 0:
                print("\nTraining interrupted. Saving checkpoint...")
                ckpt_path = ckpt_dir / f"step_{self.step:07d}_interrupted.pt"
                self.save_checkpoint(ckpt_path)
                print(f"Saved interrupted checkpoint: {ckpt_path}")
            return

        # Final checkpoint and evaluation (rank 0 only)
        if self.rank == 0:
            print("Training complete. Running final evaluation...")
            final_metrics = self.evaluate(val_loader, num_steps=euler_steps)
            # Save lightweight final checkpoint (no optimizer -- saves ~40% disk).
            # save_checkpoint also updates best_model.pt if this is the best PSNR.
            final_path = ckpt_dir / "final.pt"
            self.save_checkpoint(final_path, metrics=final_metrics, lightweight=True)
            print(
                f"Final: PSNR={final_metrics['psnr']:.2f} | "
                f"SSIM={final_metrics['ssim']:.4f} | "
                f"PCC={final_metrics['pcc']:.4f} | "
                f"LPIPS={final_metrics['lpips']:.4f}"
            )

        # Sync all ranks after training
        if self.world_size > 1:
            import torch.distributed as dist
            dist.barrier()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_sample_images(
        self,
        source: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Log per-channel source|pred|target grid to W&B.

        Creates a grid with columns:
        [source_rgb, pred_DAPI, pred_Lap2, pred_Marker,
         gt_DAPI, gt_Lap2, gt_Marker]

        Each grayscale channel is expanded to 3-channel for display.

        Args:
            source: Source images (N, C_src, H, W). Typically 4-channel.
            pred: Predicted images (N, C_out, H, W). 3-channel (DAPI, Lap2, Marker).
            target: Ground truth images (N, C_out, H, W). 3-channel.
        """
        if self.wandb_run is None:
            return

        try:
            import wandb

            # Source RGB (first 3 channels)
            src_rgb = source[:, :3].cpu()

            # Per-channel grayscale -> 3ch for display
            images = []
            for i in range(pred.shape[0]):
                row = [src_rgb[i]]
                # Predicted channels
                for c in range(pred.shape[1]):
                    row.append(pred[i, c:c+1].repeat(3, 1, 1).cpu())
                # Ground truth channels
                for c in range(target.shape[1]):
                    row.append(target[i, c:c+1].repeat(3, 1, 1).cpu())
                images.extend(row)

            ncols = 1 + pred.shape[1] + target.shape[1]  # src + pred_chs + gt_chs
            grid = make_grid(images, nrow=ncols, padding=2, normalize=False)

            ch_labels = " | ".join(self.channel_names)
            self.wandb_run.log(
                {
                    "samples": wandb.Image(
                        grid,
                        caption=(
                            f"step {self.step} | "
                            f"src | pred({ch_labels}) | gt({ch_labels})"
                        ),
                    )
                },
                step=self.step,
            )
        except Exception as e:
            logger.warning(f"Failed to log sample images: {e}")

    def _cleanup_checkpoints(self, ckpt_dir: Path) -> None:
        """Remove old checkpoints keeping only the most recent N.

        Args:
            ckpt_dir: Checkpoint directory.
        """
        keep_n = getattr(self.config, "keep_n_checkpoints", 1)
        checkpoints = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        # Exclude best_model.pt, final.pt, and interrupted checkpoints
        regular = [
            p
            for p in checkpoints
            if "best" not in p.name
            and "final" not in p.name
            and "interrupted" not in p.name
        ]

        while len(regular) > keep_n:
            oldest = regular.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")
