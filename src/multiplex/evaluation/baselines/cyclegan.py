"""CycleGAN baseline for unpaired image translation.

Simplified implementation based on Zhu et al., ICCV 2017.
Used for unpaired/pseudo-pair comparison.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaselineConfig, BaselineMethod, register_baseline
from .pix2pix import PatchDiscriminator


class ResNetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN."""

    def __init__(self, in_channels: int = 1, out_channels: int = 5, n_blocks: int = 9):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(n_blocks):
            model.append(self._resblock(in_features))

        # Upsampling
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def _resblock(self, features: int):
        return ResidualBlock(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with reflection padding."""

    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3, bias=False),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGANDiscriminator(nn.Module):
    """Discriminator that takes a single input tensor."""

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@register_baseline("cyclegan")
class CycleGANBaseline(BaselineMethod):
    """CycleGAN baseline for unpaired translation."""

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.G_AB = None  # BF -> markers
        self.G_BA = None  # markers -> BF
        self.D_A = None   # Discriminator for BF
        self.D_B = None   # Discriminator for markers
        self.lambda_cycle = config.extra.get("lambda_cycle", 10.0)
        self.lambda_identity = config.extra.get("lambda_identity", 0.5)

    def build_model(self) -> nn.Module:
        """Build CycleGAN components."""
        in_ch = self.config.extra.get("in_channels", 1)
        out_ch = self.config.extra.get("out_channels", 5)
        self.G_AB = ResNetGenerator(in_ch, out_ch).to(self.device)
        self.G_BA = ResNetGenerator(out_ch, in_ch).to(self.device)
        self.D_A = CycleGANDiscriminator(in_ch).to(self.device)
        self.D_B = CycleGANDiscriminator(out_ch).to(self.device)

        self.opt_g = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )
        self.opt_d = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )

        self.model = self.G_AB
        return self.G_AB

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one CycleGAN training step."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        # === Generator step ===
        self.opt_g.zero_grad()

        # GAN loss
        fake_B = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_g_ab = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_g_ba = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle consistency
        rec_A = self.G_BA(fake_B)
        loss_cycle_a = F.l1_loss(rec_A, real_A) * self.lambda_cycle

        rec_B = self.G_AB(fake_A)
        loss_cycle_b = F.l1_loss(rec_B, real_B) * self.lambda_cycle

        loss_g = loss_g_ab + loss_g_ba + loss_cycle_a + loss_cycle_b
        loss_g.backward()
        self.opt_g.step()

        # === Discriminator step ===
        self.opt_d.zero_grad()

        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())
        loss_d_b = (
            F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
            F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
        ) * 0.5

        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())
        loss_d_a = (
            F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
            F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
        ) * 0.5

        loss_d = loss_d_a + loss_d_b
        loss_d.backward()
        self.opt_d.step()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_cycle": (loss_cycle_a + loss_cycle_b).item(),
        }

    def train_step_amp(self, batch: Dict[str, torch.Tensor], scaler: torch.cuda.amp.GradScaler) -> Dict[str, float]:
        """Execute one CycleGAN training step with AMP."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        # === Generator step with AMP ===
        self.opt_g.zero_grad()

        with torch.cuda.amp.autocast():
            # GAN loss
            fake_B = self.G_AB(real_A)
            pred_fake_B = self.D_B(fake_B)
            loss_g_ab = F.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            fake_A = self.G_BA(real_B)
            pred_fake_A = self.D_A(fake_A)
            loss_g_ba = F.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

            # Cycle consistency
            rec_A = self.G_BA(fake_B)
            loss_cycle_a = F.l1_loss(rec_A, real_A) * self.lambda_cycle

            rec_B = self.G_AB(fake_A)
            loss_cycle_b = F.l1_loss(rec_B, real_B) * self.lambda_cycle

            loss_g = loss_g_ab + loss_g_ba + loss_cycle_a + loss_cycle_b

        scaler.scale(loss_g).backward()
        scaler.step(self.opt_g)

        # === Discriminator step with AMP ===
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast():
            pred_real_B = self.D_B(real_B)
            pred_fake_B = self.D_B(fake_B.detach())
            loss_d_b = (
                F.mse_loss(pred_real_B, torch.ones_like(pred_real_B)) +
                F.mse_loss(pred_fake_B, torch.zeros_like(pred_fake_B))
            ) * 0.5

            pred_real_A = self.D_A(real_A)
            pred_fake_A = self.D_A(fake_A.detach())
            loss_d_a = (
                F.mse_loss(pred_real_A, torch.ones_like(pred_real_A)) +
                F.mse_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
            ) * 0.5

            loss_d = loss_d_a + loss_d_b

        scaler.scale(loss_d).backward()
        scaler.step(self.opt_d)
        scaler.update()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_cycle": (loss_cycle_a + loss_cycle_b).item(),
        }

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save CycleGAN checkpoint with all models and optimizers."""
        torch.save({
            'G_AB_state_dict': self.G_AB.state_dict(),
            'G_BA_state_dict': self.G_BA.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }, path)

    def _strip_module_prefix(self, state_dict: dict) -> dict:
        """Strip 'module.' prefix from DataParallel state dicts."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_checkpoint(self, path: str) -> int:
        """Load CycleGAN checkpoint.

        Handles two checkpoint formats:
        1. Full training checkpoint with G_AB_state_dict, etc.
        2. Simplified checkpoint with model_state_dict (G_AB only)

        Also handles DataParallel checkpoints by stripping 'module.' prefix.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'G_AB_state_dict' in checkpoint:
            self.G_AB.load_state_dict(self._strip_module_prefix(checkpoint['G_AB_state_dict']))
            if 'G_BA_state_dict' in checkpoint:
                self.G_BA.load_state_dict(self._strip_module_prefix(checkpoint['G_BA_state_dict']))
            if 'D_A_state_dict' in checkpoint:
                self.D_A.load_state_dict(self._strip_module_prefix(checkpoint['D_A_state_dict']))
            if 'D_B_state_dict' in checkpoint:
                self.D_B.load_state_dict(self._strip_module_prefix(checkpoint['D_B_state_dict']))
        elif 'model_state_dict' in checkpoint:
            # Simplified format - G_AB only
            self.G_AB.load_state_dict(self._strip_module_prefix(checkpoint['model_state_dict']))
        else:
            raise KeyError(f"Checkpoint missing generator weights. Keys: {list(checkpoint.keys())}")

        if 'opt_g_state_dict' in checkpoint:
            self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        if 'opt_d_state_dict' in checkpoint:
            self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        return checkpoint.get('epoch', 0)

    def predict(self, bf: torch.Tensor) -> torch.Tensor:
        """Generate predictions."""
        self.G_AB.eval()
        with torch.no_grad():
            fake = self.G_AB(bf.to(self.device))
            return (fake + 1) / 2
