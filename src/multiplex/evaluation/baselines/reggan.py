"""RegGAN baseline implementation.

RegGAN: Registration-based GAN for image-to-image translation with misaligned data.
Paper: https://arxiv.org/abs/2110.06465

Key idea: Add a registration network to handle misaligned paired data by learning
spatial transformations that align generated images with targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .registry import BaselineConfig, register_baseline
from .cyclegan import CycleGANBaseline, ResNetGenerator, CycleGANDiscriminator


class ResidualBlock(nn.Module):
    """Residual block for registration network."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class RegistrationNetwork(nn.Module):
    """Spatial registration network that predicts deformation fields.

    Takes two images and outputs a 2D displacement field for alignment.
    Uses a U-Net style architecture with residual blocks.
    """

    def __init__(self, in_channels: int = 5, base_filters: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_filters, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_filters * 4),
            ResidualBlock(base_filters * 4),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Output: 2-channel displacement field
        self.out = nn.Conv2d(base_filters, 2, 3, padding=1)

        # Initialize output to zero (identity transformation)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Predict displacement field to align source to target.

        Args:
            source: Source image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Displacement field [B, 2, H, W]
        """
        x = torch.cat([source, target], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # Output displacement field
        flow = self.out(d1)
        return flow


class SpatialTransformer(nn.Module):
    """Differentiable spatial transformer for applying deformation fields."""

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformation.

        Args:
            img: Input image [B, C, H, W]
            flow: Displacement field [B, 2, H, W]

        Returns:
            Warped image [B, C, H, W]
        """
        B, C, H, W = img.shape

        # Create identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Add flow (normalized to [-1, 1])
        flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        new_grid = grid + flow

        # Sample
        return F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """Compute smoothness regularization for displacement field."""
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)


@register_baseline("reggan")
class RegGANBaseline(CycleGANBaseline):
    """RegGAN: CycleGAN with registration network for misaligned data.

    Extends CycleGAN by adding a registration network that learns to spatially
    align generated images with target images, handling misalignment in the training data.
    """

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        # Don't call parent __init__ directly, we'll set up differently
        self.config = config
        self.device = device
        self.model = None
        self.generator_A2B = None
        self.generator_B2A = None
        self.discriminator_A = None
        self.discriminator_B = None
        self.registration_net = None
        self.spatial_transformer = None
        self.opt_g = None
        self.opt_d = None
        self.opt_r = None
        self.scaler = None

        # Loss weights
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5
        self.lambda_reg = 1.0  # Registration loss weight
        self.lambda_smooth = 0.1  # Smoothness regularization

    def build_model(self) -> None:
        """Build RegGAN model with registration network."""
        in_ch = self.config.extra.get("in_channels", 1)
        out_ch = self.config.extra.get("out_channels", 5)

        # Generators (same as CycleGAN)
        self.generator_A2B = ResNetGenerator(
            in_channels=in_ch,
            out_channels=out_ch,
            n_blocks=6,  # Reduced for speed
        ).to(self.device)

        self.generator_B2A = ResNetGenerator(
            in_channels=out_ch,
            out_channels=in_ch,
            n_blocks=6,
        ).to(self.device)

        # Discriminators
        self.discriminator_A = CycleGANDiscriminator(
            in_channels=in_ch,
        ).to(self.device)

        self.discriminator_B = CycleGANDiscriminator(
            in_channels=out_ch,
        ).to(self.device)

        # Registration network
        self.registration_net = RegistrationNetwork(
            in_channels=out_ch,
            base_filters=32,
        ).to(self.device)

        self.spatial_transformer = SpatialTransformer().to(self.device)

        # Main model reference
        self.model = self.generator_A2B

        # Optimizers
        self.opt_g = torch.optim.Adam(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )
        self.opt_d = torch.optim.Adam(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )
        self.opt_r = torch.optim.Adam(
            self.registration_net.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )

        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with registration."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        losses = {}

        # =================== Train Generators ===================
        self.opt_g.zero_grad()
        self.opt_r.zero_grad()

        with torch.cuda.amp.autocast():
            # Forward cycle: A -> B -> A
            fake_B = self.generator_A2B(real_A)

            # Registration: align fake_B with real_B
            flow = self.registration_net(fake_B, real_B)
            fake_B_registered = self.spatial_transformer(fake_B, flow)

            rec_A = self.generator_B2A(fake_B)

            # Backward cycle: B -> A -> B
            fake_A = self.generator_B2A(real_B)
            rec_B = self.generator_A2B(fake_A)

            # Adversarial losses
            pred_fake_B = self.discriminator_B(fake_B)
            loss_g_adv_B = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            pred_fake_A = self.discriminator_A(fake_A)
            loss_g_adv_A = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle consistency losses
            loss_cycle_A = F.l1_loss(rec_A, real_A)
            loss_cycle_B = F.l1_loss(rec_B, real_B)

            # Registration loss (align generated with target)
            loss_reg = F.l1_loss(fake_B_registered, real_B)

            # Smoothness regularization on flow
            loss_smooth = smoothness_loss(flow)

            # Note: Identity loss skipped - not applicable for asymmetric channels (1->5)

            # Total generator loss
            loss_g = (
                loss_g_adv_A + loss_g_adv_B +
                self.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                self.lambda_reg * loss_reg +
                self.lambda_smooth * loss_smooth
            )

        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.opt_g)
        self.scaler.step(self.opt_r)

        # =================== Train Discriminators ===================
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast():
            # Discriminator A
            pred_real_A = self.discriminator_A(real_A)
            pred_fake_A = self.discriminator_A(fake_A.detach())
            loss_d_A = 0.5 * (
                F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
                F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
            )

            # Discriminator B
            pred_real_B = self.discriminator_B(real_B)
            pred_fake_B = self.discriminator_B(fake_B.detach())
            loss_d_B = 0.5 * (
                F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
                F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
            )

            loss_d = loss_d_A + loss_d_B

        self.scaler.scale(loss_d).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        losses = {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
            "loss_reg": loss_reg.item(),
            "loss_smooth": loss_smooth.item(),
        }

        return losses

    def train_step_amp(self, batch: Dict[str, torch.Tensor], scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        """Training step with registration using external AMP scaler."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        losses = {}

        # =================== Train Generators ===================
        self.opt_g.zero_grad()
        self.opt_r.zero_grad()

        with torch.cuda.amp.autocast():
            # Forward cycle: A -> B -> A
            fake_B = self.generator_A2B(real_A)

            # Registration: align fake_B with real_B
            flow = self.registration_net(fake_B, real_B)
            fake_B_registered = self.spatial_transformer(fake_B, flow)

            rec_A = self.generator_B2A(fake_B)

            # Backward cycle: B -> A -> B
            fake_A = self.generator_B2A(real_B)
            rec_B = self.generator_A2B(fake_A)

            # Adversarial losses
            pred_fake_B = self.discriminator_B(fake_B)
            loss_g_adv_B = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            pred_fake_A = self.discriminator_A(fake_A)
            loss_g_adv_A = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle consistency losses
            loss_cycle_A = F.l1_loss(rec_A, real_A)
            loss_cycle_B = F.l1_loss(rec_B, real_B)

            # Registration loss (align generated with target)
            loss_reg = F.l1_loss(fake_B_registered, real_B)

            # Smoothness regularization on flow
            loss_smooth = smoothness_loss(flow)

            # Note: Identity loss skipped - not applicable for asymmetric channels (1->5)

            # Total generator loss
            loss_g = (
                loss_g_adv_A + loss_g_adv_B +
                self.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                self.lambda_reg * loss_reg +
                self.lambda_smooth * loss_smooth
            )

        scaler.scale(loss_g).backward()
        scaler.step(self.opt_g)
        scaler.step(self.opt_r)

        # =================== Train Discriminators ===================
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast():
            # Discriminator A
            pred_real_A = self.discriminator_A(real_A)
            pred_fake_A = self.discriminator_A(fake_A.detach())
            loss_d_A = 0.5 * (
                F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
                F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
            )

            # Discriminator B
            pred_real_B = self.discriminator_B(real_B)
            pred_fake_B = self.discriminator_B(fake_B.detach())
            loss_d_B = 0.5 * (
                F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
                F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
            )

            loss_d = loss_d_A + loss_d_B

        scaler.scale(loss_d).backward()
        scaler.step(self.opt_d)
        scaler.update()

        losses = {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
            "loss_reg": loss_reg.item(),
            "loss_smooth": loss_smooth.item(),
        }

        return losses

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save checkpoint with all models."""
        torch.save({
            'model_state_dict': self.generator_A2B.state_dict(),
            'generator_A2B_state_dict': self.generator_A2B.state_dict(),
            'generator_B2A_state_dict': self.generator_B2A.state_dict(),
            'discriminator_A_state_dict': self.discriminator_A.state_dict(),
            'discriminator_B_state_dict': self.discriminator_B.state_dict(),
            'registration_net_state_dict': self.registration_net.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'opt_r_state_dict': self.opt_r.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }, path)

    def _strip_module_prefix(self, state_dict: dict) -> dict:
        """Strip 'module.' prefix from DataParallel state dicts."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'generator_A2B_state_dict' in checkpoint:
            self.generator_A2B.load_state_dict(self._strip_module_prefix(checkpoint['generator_A2B_state_dict']))
            if 'generator_B2A_state_dict' in checkpoint:
                self.generator_B2A.load_state_dict(self._strip_module_prefix(checkpoint['generator_B2A_state_dict']))
            if 'discriminator_A_state_dict' in checkpoint:
                self.discriminator_A.load_state_dict(self._strip_module_prefix(checkpoint['discriminator_A_state_dict']))
            if 'discriminator_B_state_dict' in checkpoint:
                self.discriminator_B.load_state_dict(self._strip_module_prefix(checkpoint['discriminator_B_state_dict']))
            if 'registration_net_state_dict' in checkpoint:
                self.registration_net.load_state_dict(self._strip_module_prefix(checkpoint['registration_net_state_dict']))
        elif 'model_state_dict' in checkpoint:
            self.generator_A2B.load_state_dict(self._strip_module_prefix(checkpoint['model_state_dict']))
        else:
            raise KeyError(f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}")

        return checkpoint.get('epoch', 0)

    def predict(self, bf: torch.Tensor) -> torch.Tensor:
        """Generate predictions.

        Args:
            bf: Input in [-1, 1] range (from dataset with normalize_to='-1_1')

        Returns:
            Output in [0, 1] range
        """
        self.generator_A2B.eval()
        with torch.no_grad():
            # Input is already in [-1, 1]
            fake = self.generator_A2B(bf.to(self.device))
            # Convert back to [0, 1]
            return (fake + 1) / 2
