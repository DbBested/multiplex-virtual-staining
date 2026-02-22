"""NICE-GAN baseline implementation.

NICE-GAN: Reusing Discriminators for Encoding: Towards Unsupervised Image-to-Image Translation
Paper: https://arxiv.org/abs/2003.00273 (CVPR 2020)

Key idea: Reuse the discriminator as an encoder, making the architecture more compact
and the encoder more informative through adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from torch.nn.utils import spectral_norm

from .registry import BaselineConfig, register_baseline


def unwrap_model(model):
    """Unwrap DataParallel to access underlying module."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


class ILN(nn.Module):
    """Instance-Layer Normalization with learnable blend."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance norm
        in_mean = x.mean(dim=[2, 3], keepdim=True)
        in_var = x.var(dim=[2, 3], keepdim=True)
        x_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        # Layer norm
        ln_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        ln_var = x.var(dim=[1, 2, 3], keepdim=True)
        x_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        # Learnable blend
        rho = torch.sigmoid(self.rho)
        out = rho * x_in + (1 - rho) * x_ln
        return out * self.gamma + self.beta


class AdaILN(nn.Module):
    """Adaptive Instance-Layer Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # Instance norm
        in_mean = x.mean(dim=[2, 3], keepdim=True)
        in_var = x.var(dim=[2, 3], keepdim=True)
        x_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        # Layer norm
        ln_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        ln_var = x.var(dim=[1, 2, 3], keepdim=True)
        x_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        # Learnable blend
        rho = torch.sigmoid(self.rho)
        out = rho * x_in + (1 - rho) * x_ln

        # Apply adaptive parameters
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return out * gamma + beta


class ResnetAdaILNBlock(nn.Module):
    """Residual block with adaptive instance-layer normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 3)
        self.norm1 = AdaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, 3)
        self.norm2 = AdaILN(dim)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return x + out


class NICEGenerator(nn.Module):
    """NICE-GAN Generator (Decoder only - encoder is in discriminator)."""

    def __init__(
        self,
        input_nc: int = 256,  # Encoded feature channels
        output_nc: int = 5,
        ngf: int = 64,
        n_blocks: int = 4,
        img_size: int = 512,
        light: bool = True,  # Light mode for speed
    ):
        super().__init__()
        self.light = light
        self.n_blocks = n_blocks

        # Calculate encoded size
        self.encoded_size = img_size // 4  # After 2 downsampling in encoder

        # FC for adaptive parameters
        if light:
            self.fc = nn.Sequential(
                nn.Linear(input_nc, input_nc),
                nn.ReLU(True),
                nn.Linear(input_nc, input_nc),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_nc * self.encoded_size * self.encoded_size, input_nc),
                nn.ReLU(True),
                nn.Linear(input_nc, input_nc),
                nn.ReLU(True),
            )

        self.gamma_fc = nn.Linear(input_nc, input_nc)
        self.beta_fc = nn.Linear(input_nc, input_nc)

        # Residual blocks with AdaILN
        self.res_blocks = nn.ModuleList([
            ResnetAdaILNBlock(input_nc) for _ in range(n_blocks)
        ])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, ngf * 2, 3),
            ILN(ngf * 2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf, 3),
            ILN(ngf),
            nn.ReLU(True),
        )

        # Output
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to image.

        Args:
            x: Encoded features [B, C, H, W]

        Returns:
            Generated image [B, output_nc, H*4, W*4]
        """
        # Get adaptive parameters
        if self.light:
            x_gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x_gap = x.view(x.size(0), -1)

        fc_out = self.fc(x_gap)
        gamma = self.gamma_fc(fc_out)
        beta = self.beta_fc(fc_out)

        # Residual blocks
        out = x
        for block in self.res_blocks:
            out = block(out, gamma, beta)

        # Upsample
        out = self.up1(out)
        out = self.up2(out)
        out = self.out(out)

        return out


class NICEDiscriminator(nn.Module):
    """NICE-GAN Discriminator that also serves as encoder.

    The encoder part is reused for the generator, making the architecture compact.
    """

    def __init__(
        self,
        input_nc: int = 1,
        ndf: int = 64,
        n_layers: int = 5,
    ):
        super().__init__()

        # Shared encoder layers (also used by generator)
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(input_nc, ndf, 3, stride=2)),
            nn.LeakyReLU(0.2, True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, stride=2)),
            nn.LeakyReLU(0.2, True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, stride=1)),
            nn.LeakyReLU(0.2, True),
        )

        # Discriminator-specific layers
        self.disc_layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, stride=1)),
            nn.LeakyReLU(0.2, True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 3, stride=1)),
        )

        # CAM (Class Activation Map) for better discrimination
        self.gap_fc = spectral_norm(nn.Linear(ndf * 4, 1, bias=False))
        self.gmp_fc = spectral_norm(nn.Linear(ndf * 4, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * 8, ndf * 4, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            out: Discrimination score
            cam_logit: CAM logit for auxiliary loss
            encoded: Encoded features (for generator)
        """
        # Encode
        encoded = self.encoder(x)

        # CAM
        gap = F.adaptive_avg_pool2d(encoded, 1).view(encoded.size(0), -1)
        gap_logit = self.gap_fc(gap)
        gap_weight = self.gap_fc.weight.unsqueeze(2).unsqueeze(3)
        gap_out = encoded * gap_weight

        gmp = F.adaptive_max_pool2d(encoded, 1).view(encoded.size(0), -1)
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = self.gmp_fc.weight.unsqueeze(2).unsqueeze(3)
        gmp_out = encoded * gmp_weight

        cam_logit = torch.cat([gap_logit, gmp_logit], dim=1)
        x_cat = torch.cat([gap_out, gmp_out], dim=1)
        x_cat = self.leaky_relu(self.conv1x1(x_cat))

        # Discriminate
        out = self.disc_layers(x_cat)

        return out, cam_logit, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input (used by generator)."""
        return self.encoder(x)


@register_baseline("nicegan")
class NICEGANBaseline:
    """NICE-GAN baseline implementation.

    Key features:
    - Discriminator is reused as encoder (no separate encoder needed)
    - Decoupled training: encoder frozen when minimizing G loss
    - Compact and efficient architecture
    """

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.generator_A2B = None
        self.generator_B2A = None
        self.discriminator_A = None
        self.discriminator_B = None
        self.opt_g = None
        self.opt_d = None
        self.scaler = None

        # Loss weights (reduced lambda_cam to prevent gradient explosion)
        self.lambda_cycle = 10.0
        self.lambda_identity = 10.0
        self.lambda_cam = 1.0  # Was 1000.0, caused NaN

    def build_model(self) -> None:
        """Build NICE-GAN model."""
        in_ch = self.config.extra.get("in_channels", 1)
        out_ch = self.config.extra.get("out_channels", 5)
        ngf = 64
        ndf = 64

        # Discriminators (also serve as encoders)
        self.discriminator_A = NICEDiscriminator(
            input_nc=in_ch,
            ndf=ndf,
        ).to(self.device)

        self.discriminator_B = NICEDiscriminator(
            input_nc=out_ch,
            ndf=ndf,
        ).to(self.device)

        # Generators (decoders only)
        self.generator_A2B = NICEGenerator(
            input_nc=ndf * 4,  # Encoded feature channels
            output_nc=out_ch,
            ngf=ngf,
            n_blocks=4,  # Reduced for speed
            light=True,
        ).to(self.device)

        self.generator_B2A = NICEGenerator(
            input_nc=ndf * 4,
            output_nc=in_ch,
            ngf=ngf,
            n_blocks=4,
            light=True,
        ).to(self.device)

        # Main model reference
        self.model = self.generator_A2B

        # Optimizers
        self.opt_g = torch.optim.Adam(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )
        self.opt_d = torch.optim.Adam(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
            lr=self.config.lr,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with decoupled encoder training."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        losses = {}

        # =================== Train Discriminators (including encoder) ===================
        self.opt_d.zero_grad()

        with torch.cuda.amp.autocast():
            # Encode (discriminator as encoder)
            _, _, enc_A = self.discriminator_A(real_A)
            _, _, enc_B = self.discriminator_B(real_B)

            # Generate
            fake_B = self.generator_A2B(enc_A)
            fake_A = self.generator_B2A(enc_B)

            # Discriminator A
            real_out_A, real_cam_A, _ = self.discriminator_A(real_A)
            fake_out_A, fake_cam_A, _ = self.discriminator_A(fake_A.detach())

            loss_d_real_A = F.mse_loss(real_out_A, torch.ones_like(real_out_A))
            loss_d_fake_A = F.mse_loss(fake_out_A, torch.zeros_like(fake_out_A))
            loss_d_cam_A = F.binary_cross_entropy_with_logits(
                real_cam_A, torch.ones_like(real_cam_A)
            ) + F.binary_cross_entropy_with_logits(
                fake_cam_A, torch.zeros_like(fake_cam_A)
            )

            # Discriminator B
            real_out_B, real_cam_B, _ = self.discriminator_B(real_B)
            fake_out_B, fake_cam_B, _ = self.discriminator_B(fake_B.detach())

            loss_d_real_B = F.mse_loss(real_out_B, torch.ones_like(real_out_B))
            loss_d_fake_B = F.mse_loss(fake_out_B, torch.zeros_like(fake_out_B))
            loss_d_cam_B = F.binary_cross_entropy_with_logits(
                real_cam_B, torch.ones_like(real_cam_B)
            ) + F.binary_cross_entropy_with_logits(
                fake_cam_B, torch.zeros_like(fake_cam_B)
            )

            loss_d = (
                loss_d_real_A + loss_d_fake_A + loss_d_real_B + loss_d_fake_B +
                self.lambda_cam * (loss_d_cam_A + loss_d_cam_B)
            )

        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.opt_d)
        torch.nn.utils.clip_grad_norm_(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
            max_norm=1.0
        )
        self.scaler.step(self.opt_d)

        # =================== Train Generators (encoder frozen) ===================
        self.opt_g.zero_grad()

        # Freeze encoder parts of discriminators (unwrap DataParallel if needed)
        dis_A = unwrap_model(self.discriminator_A)
        dis_B = unwrap_model(self.discriminator_B)
        for param in dis_A.encoder.parameters():
            param.requires_grad = False
        for param in dis_B.encoder.parameters():
            param.requires_grad = False

        with torch.cuda.amp.autocast():
            # Encode with frozen encoder (use unwrapped models)
            enc_A = dis_A.encode(real_A)
            enc_B = dis_B.encode(real_B)

            # Generate
            fake_B = self.generator_A2B(enc_A)
            fake_A = self.generator_B2A(enc_B)

            # Cycle: encode fake, decode back
            _, _, enc_fake_B = self.discriminator_B(fake_B)
            _, _, enc_fake_A = self.discriminator_A(fake_A)
            rec_A = self.generator_B2A(enc_fake_B)
            rec_B = self.generator_A2B(enc_fake_A)

            # Adversarial losses
            fake_out_B, fake_cam_B, _ = self.discriminator_B(fake_B)
            fake_out_A, fake_cam_A, _ = self.discriminator_A(fake_A)

            loss_g_adv_B = F.mse_loss(fake_out_B, torch.ones_like(fake_out_B))
            loss_g_adv_A = F.mse_loss(fake_out_A, torch.ones_like(fake_out_A))
            loss_g_cam_B = F.binary_cross_entropy_with_logits(
                fake_cam_B, torch.ones_like(fake_cam_B)
            )
            loss_g_cam_A = F.binary_cross_entropy_with_logits(
                fake_cam_A, torch.ones_like(fake_cam_A)
            )

            # Cycle consistency
            loss_cycle_A = F.l1_loss(rec_A, real_A)
            loss_cycle_B = F.l1_loss(rec_B, real_B)

            # Identity loss
            idt_B = self.generator_A2B(enc_B)
            idt_A = self.generator_B2A(enc_A)
            loss_idt_A = F.l1_loss(idt_A, real_A)
            loss_idt_B = F.l1_loss(idt_B, real_B)

            loss_g = (
                loss_g_adv_A + loss_g_adv_B +
                self.lambda_cam * (loss_g_cam_A + loss_g_cam_B) +
                self.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                self.lambda_identity * (loss_idt_A + loss_idt_B)
            )

        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.opt_g)
        torch.nn.utils.clip_grad_norm_(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            max_norm=1.0
        )
        self.scaler.step(self.opt_g)
        self.scaler.update()

        # Unfreeze encoder (use already unwrapped references)
        for param in dis_A.encoder.parameters():
            param.requires_grad = True
        for param in dis_B.encoder.parameters():
            param.requires_grad = True

        losses = {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item(),
            "loss_cam": (loss_g_cam_A + loss_g_cam_B).item(),
        }

        return losses

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save checkpoint."""
        torch.save({
            'model_state_dict': self.generator_A2B.state_dict(),
            'generator_A2B_state_dict': self.generator_A2B.state_dict(),
            'generator_B2A_state_dict': self.generator_B2A.state_dict(),
            'discriminator_A_state_dict': self.discriminator_A.state_dict(),
            'discriminator_B_state_dict': self.discriminator_B.state_dict(),
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
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if 'generator_A2B_state_dict' in checkpoint:
            self.generator_A2B.load_state_dict(
                self._strip_module_prefix(checkpoint['generator_A2B_state_dict'])
            )
            if 'generator_B2A_state_dict' in checkpoint:
                self.generator_B2A.load_state_dict(
                    self._strip_module_prefix(checkpoint['generator_B2A_state_dict'])
                )
            if 'discriminator_A_state_dict' in checkpoint:
                self.discriminator_A.load_state_dict(
                    self._strip_module_prefix(checkpoint['discriminator_A_state_dict'])
                )
            if 'discriminator_B_state_dict' in checkpoint:
                self.discriminator_B.load_state_dict(
                    self._strip_module_prefix(checkpoint['discriminator_B_state_dict'])
                )
        elif 'model_state_dict' in checkpoint:
            self.generator_A2B.load_state_dict(
                self._strip_module_prefix(checkpoint['model_state_dict'])
            )
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
        dis_A = unwrap_model(self.discriminator_A)
        gen_A2B = unwrap_model(self.generator_A2B)
        dis_A.eval()
        gen_A2B.eval()

        with torch.no_grad():
            # Input is already in [-1, 1]
            encoded = dis_A.encode(bf.to(self.device))
            # Decode
            fake = gen_A2B(encoded)
            # Convert back to [0, 1]
            return (fake + 1) / 2
