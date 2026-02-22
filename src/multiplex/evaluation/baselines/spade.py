"""SPADE baseline for semantic-aware image synthesis.

Simplified implementation based on Park et al., CVPR 2019.
Uses spatially-adaptive normalization for semantic-guided synthesis.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaselineConfig, BaselineMethod, register_baseline


class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization layer.

    Modulates normalized activations using learned scale and bias
    derived from a semantic/conditioning input.
    """

    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int = 128):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # Shared convolution for conditioning
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Separate convolutions for gamma and beta
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, 3, 1, 1)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, 3, 1, 1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        # Normalize
        normalized = self.param_free_norm(x)

        # Resize segmap to match x
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='nearest')

        # Compute modulation parameters
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply modulation
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization."""

    def __init__(self, in_nc: int, out_nc: int, label_nc: int):
        super().__init__()

        middle_nc = min(in_nc, out_nc)

        self.learned_shortcut = (in_nc != out_nc)

        self.conv_0 = nn.Conv2d(in_nc, middle_nc, 3, 1, 1)
        self.conv_1 = nn.Conv2d(middle_nc, out_nc, 3, 1, 1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_nc, out_nc, 1, bias=False)

        self.norm_0 = SPADE(in_nc, label_nc)
        self.norm_1 = SPADE(middle_nc, label_nc)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_nc, label_nc)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2))

        return x_s + dx

    def shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s


class SPADEGenerator(nn.Module):
    """SPADE-based generator for semantic image synthesis.

    Uses brightfield input as semantic conditioning to generate markers.
    """

    def __init__(self, label_nc: int = 1, out_channels: int = 5, ngf: int = 64):
        super().__init__()

        self.ngf = ngf
        nf = ngf

        # Initial fully-connected layer to create feature map from noise
        self.fc = nn.Linear(256, 16 * nf * 4 * 4)

        # SPADE residual blocks (progressively upsample)
        self.head = SPADEResBlock(16 * nf, 16 * nf, label_nc)
        self.up_1 = SPADEResBlock(16 * nf, 8 * nf, label_nc)
        self.up_2 = SPADEResBlock(8 * nf, 4 * nf, label_nc)
        self.up_3 = SPADEResBlock(4 * nf, 2 * nf, label_nc)
        self.up_4 = SPADEResBlock(2 * nf, 1 * nf, label_nc)

        # Final output convolution
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, seg: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            seg: Semantic/conditioning input (brightfield) of shape (B, 1, H, W)
            z: Optional noise vector of shape (B, 256). If None, uses zeros.

        Returns:
            Generated output of shape (B, out_channels, H, W)
        """
        B = seg.shape[0]

        if z is None:
            z = torch.zeros(B, 256, device=seg.device)

        # Create initial feature map from noise
        x = self.fc(z)
        x = x.view(B, 16 * self.ngf, 4, 4)

        # Progressive upsampling with SPADE
        x = self.head(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        x = self.up(x)
        x = self.up_4(x, seg)
        x = self.up(x)

        # Final output
        x = self.conv_out(x)

        # Resize to match input
        x = F.interpolate(x, size=seg.shape[2:], mode='bilinear', align_corners=False)

        return x


class SPADEDiscriminator(nn.Module):
    """Multi-scale discriminator for SPADE."""

    def __init__(self, in_channels: int = 6, ndf: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))


@register_baseline("spade")
class SPADEBaseline(BaselineMethod):
    """SPADE baseline for semantic-aware marker synthesis.

    Uses brightfield as conditioning to spatially modulate
    the generation of virtual IHC markers.
    """

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.generator = None
        self.discriminator = None
        self.opt_g = None
        self.opt_d = None
        self.lambda_feat = config.extra.get("lambda_feat", 10.0)
        self.lambda_vgg = config.extra.get("lambda_vgg", 10.0)

    def build_model(self) -> nn.Module:
        """Build SPADE generator and discriminator."""
        self.generator = SPADEGenerator(
            label_nc=1,
            out_channels=5,
            ngf=64,
        ).to(self.device)

        self.discriminator = SPADEDiscriminator(
            in_channels=6,  # BF + markers
            ndf=64,
        ).to(self.device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(0.0, 0.999),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(0.0, 0.999),
        )

        self.model = self.generator
        return self.generator

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one SPADE training step."""
        bf = batch["bf"].to(self.device)
        markers = batch["markers"].to(self.device)

        # Normalize to [-1, 1]
        markers_norm = markers * 2 - 1

        # === Discriminator step ===
        self.opt_d.zero_grad()

        fake = self.generator(bf)

        pred_real = self.discriminator(bf, markers_norm)
        pred_fake = self.discriminator(bf, fake.detach())

        # Hinge loss
        loss_d_real = F.relu(1 - pred_real).mean()
        loss_d_fake = F.relu(1 + pred_fake).mean()
        loss_d = loss_d_real + loss_d_fake

        loss_d.backward()
        self.opt_d.step()

        # === Generator step ===
        self.opt_g.zero_grad()

        fake = self.generator(bf)
        pred_fake = self.discriminator(bf, fake)

        # Hinge loss for generator
        loss_g_gan = -pred_fake.mean()

        # Feature matching loss (L1 on intermediate features)
        loss_g_feat = F.l1_loss(fake, markers_norm) * self.lambda_feat

        loss_g = loss_g_gan + loss_g_feat

        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_g_gan": loss_g_gan.item(),
            "loss_g_feat": loss_g_feat.item(),
        }

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save SPADE checkpoint with all models and optimizers."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load SPADE checkpoint.

        Handles two checkpoint formats:
        1. Full training checkpoint with generator_state_dict/discriminator_state_dict
        2. Simplified checkpoint with model_state_dict (generator only)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Simplified format - generator only
            self.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError(f"Checkpoint missing generator weights. Keys: {list(checkpoint.keys())}")

        if 'opt_g_state_dict' in checkpoint:
            self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        if 'opt_d_state_dict' in checkpoint:
            self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        return checkpoint.get('epoch', 0)

    def predict(self, bf: torch.Tensor) -> torch.Tensor:
        """Generate predictions."""
        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(bf.to(self.device))
            # Convert from [-1, 1] to [0, 1]
            return (fake + 1) / 2
