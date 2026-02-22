"""pix2pix baseline for paired image translation.

Simplified implementation based on the official pix2pix architecture
(Isola et al., CVPR 2017) for baseline comparison.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaselineConfig, BaselineMethod, register_baseline


class UNetGenerator(nn.Module):
    """Simple U-Net generator for pix2pix baseline.

    Follows pix2pix architecture with skip connections.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 5):
        super().__init__()

        # Encoder
        self.enc1 = self._encoder_block(in_channels, 64, normalize=False)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        self.enc5 = self._encoder_block(512, 512)

        # Decoder with skip connections
        self.dec5 = self._decoder_block(512, 512, dropout=True)
        self.dec4 = self._decoder_block(1024, 256, dropout=True)  # 512 + 512
        self.dec3 = self._decoder_block(512, 128)  # 256 + 256
        self.dec2 = self._decoder_block(256, 64)   # 128 + 128
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 64 + 64
            nn.Tanh(),
        )

    def _encoder_block(self, in_ch: int, out_ch: int, normalize: bool = True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_ch: int, out_ch: int, dropout: bool = False):
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.insert(2, nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with skip connections
        d5 = self.dec5(e5)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return d1


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_channels: int = 6):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))


@register_baseline("pix2pix")
class Pix2PixBaseline(BaselineMethod):
    """pix2pix baseline for paired BF -> marker translation."""

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.generator = None
        self.discriminator = None
        self.opt_g = None
        self.opt_d = None
        self.lambda_l1 = config.extra.get("lambda_l1", 100.0)

    def build_model(self) -> nn.Module:
        """Build generator and discriminator."""
        in_ch = self.config.extra.get("in_channels", 1)
        out_ch = self.config.extra.get("out_channels", 5)
        self.generator = UNetGenerator(in_channels=in_ch, out_channels=out_ch).to(self.device)
        self.discriminator = PatchDiscriminator(in_channels=in_ch + out_ch).to(self.device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999),
        )

        self.model = self.generator
        return self.generator

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one pix2pix training step."""
        bf = batch["bf"].to(self.device)
        markers = batch["markers"].to(self.device)

        # Normalize markers to [-1, 1] (generator uses Tanh)
        markers_norm = markers * 2 - 1

        # === Discriminator step ===
        self.opt_d.zero_grad()

        fake = self.generator(bf)
        pred_real = self.discriminator(bf, markers_norm)
        pred_fake = self.discriminator(bf, fake.detach())

        loss_d_real = F.binary_cross_entropy_with_logits(
            pred_real, torch.ones_like(pred_real)
        )
        loss_d_fake = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake)
        )
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        loss_d.backward()
        self.opt_d.step()

        # === Generator step ===
        self.opt_g.zero_grad()

        pred_fake_g = self.discriminator(bf, fake)
        loss_g_gan = F.binary_cross_entropy_with_logits(
            pred_fake_g, torch.ones_like(pred_fake_g)
        )
        loss_g_l1 = F.l1_loss(fake, markers_norm) * self.lambda_l1
        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_g_gan": loss_g_gan.item(),
            "loss_g_l1": loss_g_l1.item(),
        }

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save pix2pix checkpoint with all models and optimizers."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load pix2pix checkpoint.

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


@register_baseline("pix2pix_hd")
class Pix2PixHDBaseline(Pix2PixBaseline):
    """pix2pix-HD variant with multi-scale discriminator (placeholder).

    For now, uses same architecture as pix2pix.
    Can be extended with multi-scale discriminator if needed.
    """
    pass
