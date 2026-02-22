"""Flow matching loss and ODE samplers for ConditionalJiT.

Timestep convention: t=0 is pure noise, t=1 is clean data
Interpolant: z_t = t * x1 + (1 - t) * x0, where x0 ~ N(0,I), x1 = clean target
Velocity: dz/dt = x1 - x0 (constant along straight paths)
ODE integration: from t=0 to t=1 (noise to clean)

This module provides:
- sample_timesteps_logit_normal: Logit-normal timestep distribution (SD3)
- FlowMatchingLoss: Velocity or x-prediction flow matching loss (configurable L1/MSE)
- euler_sample: First-order Euler ODE sampler
- heun_sample: Second-order Heun (predictor-corrector) ODE sampler
- direct_predict: Single-step x-prediction (bypasses ODE integration)

Reference:
    SiT: Exploring Flow and Diffusion-based Generative Models with Scalable
         Interpolant Transformers (github.com/willisma/SiT)
    SD3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
         (arXiv:2403.03206)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device | str,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps from logit-normal distribution.

    logit-normal(m, s): t = sigmoid(m + s * z), z ~ N(0,1)

    With m=0, s=1 (SD3 default): concentrates mass on mid-range t values,
    less training on t near 0 or 1 (where the task is trivial).

    Args:
        batch_size: Number of timesteps to sample.
        device: Device for output tensor.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.

    Returns:
        Timesteps of shape (B,) in [1e-5, 1 - 1e-5].
    """
    u = torch.randn(batch_size, device=device)
    t = torch.sigmoid(mean + std * u)
    t = t.clamp(1e-5, 1 - 1e-5)
    return t


class FlowMatchingLoss(nn.Module):
    """Flow matching loss with velocity or x-prediction.

    Supports two prediction modes:
    - "velocity": Model predicts v = x1 - x0. Loss = MSE(v_pred, x1 - noise).
      This is the standard modern approach (SiT, SD3).
    - "x_prediction": Model predicts x1 directly. Loss is configurable:
      MSE (default, backward compatible) or L1 (v6.0 fix for dynamic
      range compression). L1 produces conditional median instead of mean,
      preserving bimodal intensity distributions (e.g., bright nuclei on
      dark background).

    Supports two flow types:
    - Standard: Interpolates between noise and target.
      z_t = (1-t)*noise + t*target
    - Bridge: Interpolates between source projection and target.
      z_t = (1-t)*source_proj + t*target + sigma*sqrt(t*(1-t))*noise
      Much easier task for paired I2I since the starting point has
      correlated spatial structure with the target.

    Both modes also return the predicted clean image x1_hat for future
    auxiliary loss integration (Phase 24 BioLoss, LPIPS, etc.).

    Args:
        prediction_type: Either "velocity" or "x_prediction".
        bridge: Whether to use bridge matching (source→target flow).
        bridge_sigma: Noise scale for bridge matching. Default 0.1.
        loss_type: Loss function for x_prediction mode. Either "mse"
            (default) or "l1". Velocity mode always uses MSE regardless
            of this setting.

    Example:
        >>> loss_fn = FlowMatchingLoss(prediction_type="x_prediction", bridge=True)
        >>> result = loss_fn(model, source, target, avail_config)
        >>> result["loss"].backward()
    """

    def __init__(
        self,
        prediction_type: str = "velocity",
        bridge: bool = False,
        bridge_sigma: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__()
        if prediction_type not in ("velocity", "x_prediction"):
            raise ValueError(
                f"prediction_type must be 'velocity' or 'x_prediction', "
                f"got '{prediction_type}'"
            )
        valid_loss_types = ("mse", "l1", "log_mse", "hybrid_mse", "weighted_mse", "direct_l1")
        if loss_type not in valid_loss_types:
            raise ValueError(
                f"loss_type must be one of {valid_loss_types}, got '{loss_type}'"
            )
        self.prediction_type = prediction_type
        self.bridge = bridge
        self.bridge_sigma = bridge_sigma
        self.loss_type = loss_type

    def forward(
        self,
        model: nn.Module,
        source: torch.Tensor,
        target: torch.Tensor,
        avail_config: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute flow matching loss.

        Args:
            model: ConditionalJiT model. Takes (z_t, t, source, avail_config)
                and returns prediction of shape (B, C, H, W).
            source: Source image of shape (B, C_src, H, W).
            target: Clean target image of shape (B, C_out, H, W) in [0, 1].
            avail_config: Availability config of shape (B,).

        Returns:
            Dict with keys:
            - "loss": Scalar MSE loss.
            - "x1_hat": Non-detached predicted clean image of shape
              (B, C_out, H, W) for auxiliary loss computation (gradient
              flows back to model).
            - "x1_hat_vis": Detached copy for logging/visualization (no gradient).
            - "timesteps": Sampled timesteps of shape (B,).
        """
        B = target.shape[0]
        device = target.device

        # Direct L1: no flow matching, fixed t=0.5, zeros as z_t input.
        # Model must reconstruct target from source alone (via cross-attention).
        if self.loss_type == "direct_l1":
            t = torch.full((B,), 0.5, device=device)
            z_t = torch.zeros_like(target)
            pred = model(z_t, t, source, avail_config)
            loss = F.l1_loss(pred, target)
            return {
                "loss": loss,
                "x1_hat": pred,
                "x1_hat_vis": pred.detach(),
                "timesteps": t,
            }

        # Sample timesteps from logit-normal distribution
        t = sample_timesteps_logit_normal(B, device)
        t_broadcast = t.view(B, 1, 1, 1)

        noise = torch.randn_like(target)

        if self.bridge and hasattr(model, 'get_bridge_start'):
            # Bridge matching: interpolate source_proj → target
            # Access unwrapped model for DDP
            model_unwrapped = model.module if hasattr(model, 'module') else model
            x_0 = model_unwrapped.get_bridge_start(source)

            # Bridge interpolation with stochastic noise
            # z_t = (1-t)*x_0 + t*x_1 + sigma*sqrt(t*(1-t))*eps
            sigma_t = self.bridge_sigma * torch.sqrt(
                t_broadcast * (1 - t_broadcast)
            )
            z_t = (1 - t_broadcast) * x_0 + t_broadcast * target + sigma_t * noise
        else:
            # Standard flow matching: interpolate noise → target
            x_0 = noise
            z_t = t_broadcast * target + (1 - t_broadcast) * noise

        # Model prediction
        pred = model(z_t, t, source, avail_config)

        if self.prediction_type == "velocity":
            # Velocity target: dz/dt = x1 - x0
            velocity_target = target - x_0
            loss = F.mse_loss(pred, velocity_target)
            # Recover predicted clean image: x1_hat = z_t + (1 - t) * v_pred
            x1_hat = z_t + (1 - t_broadcast) * pred
        else:
            # X-prediction target: x1 (configurable L1, MSE, or log-MSE)
            if self.loss_type == "l1":
                loss = F.l1_loss(pred, target)
            elif self.loss_type == "log_mse":
                # Log-space MSE: equalizes gradient across full dynamic range.
                # Prevents mean-prediction collapse on sparse fluorescence data
                # where MSE/L1 converge to near-zero (the conditional mean/median).
                eps = 1e-3
                log_pred = torch.log(pred.clamp(min=eps) + eps)
                log_target = torch.log(target.clamp(min=eps) + eps)
                loss = F.mse_loss(log_pred, log_target)
            elif self.loss_type == "hybrid_mse":
                # Hybrid: MSE preserves Lap2/Marker (genuinely sparse channels),
                # log-MSE boosts DAPI dynamic range. Best of both worlds.
                eps = 1e-3
                mse_loss = F.mse_loss(pred, target)
                log_pred = torch.log(pred.clamp(min=eps) + eps)
                log_target = torch.log(target.clamp(min=eps) + eps)
                log_loss = F.mse_loss(log_pred, log_target)
                loss = mse_loss + 0.1 * log_loss
            elif self.loss_type == "weighted_mse":
                # Intensity-weighted MSE: bright target pixels get higher weight.
                # Prevents mean-prediction collapse by amplifying gradients for
                # non-zero regions without distorting near-zero predictions
                # (unlike log_mse which over-penalizes near-zero).
                beta = 10.0
                weight = 1.0 + beta * target.detach()
                loss = (weight * (pred - target) ** 2).mean()
            else:
                loss = F.mse_loss(pred, target)
            x1_hat = pred

        return {
            "loss": loss,
            "x1_hat": x1_hat,              # Non-detached for auxiliary losses (BioLoss)
            "x1_hat_vis": x1_hat.detach(), # Detached for logging/visualization
            "timesteps": t,
        }


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    source: torch.Tensor,
    avail_config: torch.Tensor,
    num_steps: int = 50,
    img_size: int = 512,
    out_chans: int = 3,
    prediction_type: str = "velocity",
) -> torch.Tensor:
    """Euler ODE sampler for flow matching (1st order).

    Integrates dz/dt from t=0 (noise) to t=1 (clean) using uniform
    step size dt = 1/num_steps.

    Supports two prediction modes:
    - "velocity": Model directly outputs velocity v. Step: z = z + v * dt.
    - "x_prediction": Model outputs predicted clean image x1_hat.
      Velocity derived as v = (x1_hat - z) / (1 - t). Step: z = z + v * dt.

    Args:
        model: ConditionalJiT model.
        source: Source image of shape (B, C_src, H, W).
        avail_config: Availability config of shape (B,).
        num_steps: Number of integration steps. Default 50.
        img_size: Spatial size of the output image. Default 512.
        out_chans: Number of output channels. Default 3.
        prediction_type: "velocity" or "x_prediction". Default "velocity".

    Returns:
        Generated image of shape (B, out_chans, img_size, img_size)
        clamped to [0, 1].
    """
    B = source.shape[0]
    device = source.device

    # Start from pure noise
    z = torch.randn(B, out_chans, img_size, img_size, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_scalar = i / num_steps
        t = torch.full((B,), t_scalar, device=device)
        pred = model(z, t, source, avail_config)

        if prediction_type == "x_prediction":
            # Derive velocity from x-prediction: v = (x1_hat - z) / (1 - t)
            v = (pred - z) / (1 - t_scalar + 1e-6)
        else:
            v = pred

        z = z + v * dt

    return z.clamp(0, 1)


@torch.no_grad()
def bridge_sample(
    model: nn.Module,
    source: torch.Tensor,
    avail_config: torch.Tensor,
    num_steps: int = 50,
    prediction_type: str = "x_prediction",
) -> torch.Tensor:
    """Bridge ODE sampler: starts from source projection, not noise.

    For bridge matching, the ODE starts from source_proj (a learned
    projection of the source image to target space) and integrates
    toward the target. This produces sharper outputs because the
    starting point has correlated spatial structure with the target.

    Args:
        model: ConditionalJiT model with use_bridge=True.
        source: Source image of shape (B, C_src, H, W).
        avail_config: Availability config of shape (B,).
        num_steps: Number of integration steps. Default 50.
        prediction_type: "velocity" or "x_prediction". Default "x_prediction".

    Returns:
        Generated image of shape (B, out_chans, img_size, img_size)
        clamped to [0, 1].
    """
    B = source.shape[0]
    device = source.device

    # Start from source projection
    model_unwrapped = model.module if hasattr(model, 'module') else model
    z = model_unwrapped.get_bridge_start(source)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_scalar = i / num_steps
        t = torch.full((B,), t_scalar, device=device)
        pred = model(z, t, source, avail_config)

        if prediction_type == "x_prediction":
            v = (pred - z) / (1 - t_scalar + 1e-6)
        else:
            v = pred

        z = z + v * dt

    return z.clamp(0, 1)


@torch.no_grad()
def heun_sample(
    model: nn.Module,
    source: torch.Tensor,
    avail_config: torch.Tensor,
    num_steps: int = 25,
    img_size: int = 512,
    out_chans: int = 3,
    prediction_type: str = "velocity",
) -> torch.Tensor:
    """Heun ODE sampler for flow matching (2nd order, predictor-corrector).

    Uses Heun's method: evaluate velocity at current point (predictor),
    take tentative Euler step, evaluate velocity at predicted point
    (corrector), average both velocities for the final step.

    Costs 2 NFE (neural function evaluations) per step, but achieves
    2nd-order accuracy. Heun with N steps roughly matches Euler with 2N
    steps in quality.

    Supports both "velocity" and "x_prediction" modes (see euler_sample).

    Args:
        model: ConditionalJiT model.
        source: Source image of shape (B, C_src, H, W).
        avail_config: Availability config of shape (B,).
        num_steps: Number of integration steps. Default 25.
        img_size: Spatial size of the output image. Default 512.
        out_chans: Number of output channels. Default 3.
        prediction_type: "velocity" or "x_prediction". Default "velocity".

    Returns:
        Generated image of shape (B, out_chans, img_size, img_size)
        clamped to [0, 1].
    """
    B = source.shape[0]
    device = source.device

    def _to_velocity(pred, z_cur, t_scalar):
        if prediction_type == "x_prediction":
            return (pred - z_cur) / (1 - t_scalar + 1e-6)
        return pred

    # Start from pure noise
    z = torch.randn(B, out_chans, img_size, img_size, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_cur_scalar = i / num_steps
        t_cur = torch.full((B,), t_cur_scalar, device=device)
        t_next_scalar = min((i + 1) / num_steps, 1.0 - 1e-5)
        t_next = torch.full((B,), t_next_scalar, device=device)

        # Predictor: evaluate velocity at current point
        pred1 = model(z, t_cur, source, avail_config)
        v1 = _to_velocity(pred1, z, t_cur_scalar)
        z_pred = z + v1 * dt

        if i < num_steps - 1:
            # Corrector: evaluate velocity at predicted point
            pred2 = model(z_pred, t_next, source, avail_config)
            v2 = _to_velocity(pred2, z_pred, t_next_scalar)
            # Average both velocities for 2nd-order accuracy
            z = z + 0.5 * (v1 + v2) * dt
        else:
            # Last step: just use Euler (no corrector needed)
            z = z_pred

    return z.clamp(0, 1)


@torch.no_grad()
def direct_predict(
    model: nn.Module,
    source: torch.Tensor,
    avail_config: torch.Tensor,
    num_steps: int = 1,       # Ignored, kept for API compatibility
    img_size: int = 512,
    out_chans: int = 3,
    prediction_type: str = "x_prediction",
    timestep: float = 0.5,    # Fixed timestep for the single forward pass
) -> torch.Tensor:
    """Single-step direct prediction, bypassing ODE integration entirely.

    For x-prediction models that learned f(z_t, t, source) ~= E[x1|source],
    this recovers training-time quality (~20 dB PSNR) by querying the model
    at a fixed timestep with random noise input, then treating the output
    as the predicted clean image.

    This exists because Phase 27 diagnosis revealed that the x-prediction
    model outputs near-constant mean predictions regardless of z_t (~15%
    variation). This minimizes training MSE but completely breaks multi-step
    ODE integration (Euler/Heun produce near-black outputs). A single forward
    pass at the logit-normal mode (t=0.5) recovers the model's learned mean
    prediction directly.

    The ``timestep`` parameter controls at what t the model is queried. The
    default 0.5 is the logit-normal mode where most training data lives.
    Since the model largely ignores z_t, the noise input has minimal effect
    on the output.

    The ``num_steps`` parameter is accepted but ignored, for API compatibility
    with ``euler_sample`` and ``heun_sample``.

    Args:
        model: ConditionalJiT model.
        source: Source image of shape (B, C_src, H, W).
        avail_config: Availability config of shape (B,).
        num_steps: Ignored. Kept for API compatibility.
        img_size: Spatial size of the output image. Default 512.
        out_chans: Number of output channels. Default 3.
        prediction_type: "velocity" or "x_prediction". Default "x_prediction".
        timestep: Fixed timestep for the forward pass. Default 0.5.

    Returns:
        Generated image of shape (B, out_chans, img_size, img_size)
        clamped to [0, 1].
    """
    B = source.shape[0]
    device = source.device

    # Start from random noise (model largely ignores this)
    z = torch.randn(B, out_chans, img_size, img_size, device=device)

    # Fixed timestep for the single forward pass
    t = torch.full((B,), timestep, device=device)

    # Single forward pass
    pred = model(z, t, source, avail_config)

    if prediction_type == "x_prediction":
        # Model directly predicts clean image
        x1_hat = pred
    else:
        # Velocity mode: recover x1 from velocity prediction
        # x1 = z_t + (1 - t) * v
        x1_hat = z + (1 - timestep) * pred

    return x1_hat.clamp(0, 1)
