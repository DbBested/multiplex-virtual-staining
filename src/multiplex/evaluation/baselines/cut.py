"""CUT (Contrastive Unpaired Translation) baseline.

Simplified implementation based on Park et al., ECCV 2020.
Uses PatchNCE loss for unpaired training.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BaselineConfig, BaselineMethod, register_baseline
from .cyclegan import ResNetGenerator, CycleGANDiscriminator


def unwrap_model(model):
    """Unwrap DataParallel to access underlying module."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


class PatchSampleMLP(nn.Module):
    """MLP for projecting features to NCE space."""

    def __init__(self, in_features: int, out_features: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@register_baseline("cut")
class CUTBaseline(BaselineMethod):
    """CUT baseline using contrastive learning."""

    def __init__(self, config: BaselineConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.generator = None
        self.discriminator = None
        self.nce_layers = config.extra.get("nce_layers", [0, 4, 8, 12, 16])
        self.nce_t = config.extra.get("nce_t", 0.07)
        self.lambda_nce = config.extra.get("lambda_nce", 1.0)
        self.mlps = None

    def build_model(self) -> nn.Module:
        """Build CUT components."""
        in_ch = self.config.extra.get("in_channels", 1)
        out_ch = self.config.extra.get("out_channels", 5)
        self.generator = ResNetGenerator(in_ch, out_ch).to(self.device)
        self.discriminator = CycleGANDiscriminator(out_ch).to(self.device)

        # Create MLPs for each layer
        self.mlps = nn.ModuleList()
        sample_input = torch.randn(1, in_ch, 256, 256, device=self.device)

        # Get feature dimensions at each layer
        x = sample_input
        layer_idx = 0
        for i, layer in enumerate(self.generator.model):
            x = layer(x)
            if layer_idx < len(self.nce_layers) and i == self.nce_layers[layer_idx]:
                feat_dim = x.shape[1]
                self.mlps.append(PatchSampleMLP(feat_dim, 256).to(self.device))
                layer_idx += 1

        # If we didn't find enough layers, add MLPs for remaining specified layers
        while len(self.mlps) < len(self.nce_layers):
            self.mlps.append(PatchSampleMLP(64, 256).to(self.device))

        self.opt_g = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.mlps.parameters()),
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

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at NCE layers."""
        features = []
        layer_idx = 0
        gen = unwrap_model(self.generator)
        for i, layer in enumerate(gen.model):
            x = layer(x)
            if layer_idx < len(self.nce_layers) and i == self.nce_layers[layer_idx]:
                features.append(x)
                layer_idx += 1
        return features

    def _patchnce_loss(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        mlp: nn.Module,
    ) -> torch.Tensor:
        """Compute PatchNCE loss between query and key features."""
        B, C, H, W = feat_q.shape
        feat_q = feat_q.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        feat_k = feat_k.flatten(2).permute(0, 2, 1)

        # Sample patches
        num_patches = min(256, H * W)
        indices = torch.randperm(H * W, device=feat_q.device)[:num_patches]

        feat_q = feat_q[:, indices, :]  # (B, num_patches, C)
        feat_k = feat_k[:, indices, :]

        # Project through MLP
        feat_q = mlp(feat_q)  # (B, num_patches, 256)
        feat_k = mlp(feat_k)

        # Normalize
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)

        # InfoNCE loss
        logits = torch.bmm(feat_q, feat_k.transpose(1, 2)) / self.nce_t
        labels = torch.arange(num_patches, device=logits.device).unsqueeze(0).expand(B, -1)

        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one CUT training step."""
        # Data is already in [-1, 1] from dataset with normalize_to='-1_1'
        real_A = batch["bf"].to(self.device)
        real_B = batch["markers"].to(self.device)

        # === Generator step ===
        self.opt_g.zero_grad()

        # Forward pass with feature extraction from source
        feat_A = self._extract_features(real_A)
        fake_B = self.generator(real_A)

        # GAN loss
        pred_fake = self.discriminator(fake_B)
        loss_g_gan = F.mse_loss(pred_fake, torch.ones_like(pred_fake))

        # PatchNCE loss - identity NCE between source features
        # This encourages the generator to preserve spatial structure
        # We use a slightly different sample of patches as "query" vs "key"
        loss_nce = 0
        num_valid_layers = len(feat_A)
        if num_valid_layers > 0 and len(self.mlps) > 0:
            for i in range(min(num_valid_layers, len(self.mlps))):
                # Use same features with different patch samples for identity NCE
                loss_nce += self._patchnce_loss(feat_A[i], feat_A[i].detach(), self.mlps[i])
            loss_nce = loss_nce / min(num_valid_layers, len(self.mlps)) * self.lambda_nce
        else:
            loss_nce = torch.tensor(0.0, device=self.device)

        loss_g = loss_g_gan + loss_nce
        loss_g.backward()
        self.opt_g.step()

        # === Discriminator step ===
        self.opt_d.zero_grad()

        pred_real = self.discriminator(real_B)
        pred_fake = self.discriminator(fake_B.detach())

        loss_d = (
            F.mse_loss(pred_real, torch.ones_like(pred_real)) +
            F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        ) * 0.5

        loss_d.backward()
        self.opt_d.step()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_nce": loss_nce.item() if isinstance(loss_nce, torch.Tensor) else loss_nce,
            "loss_g_gan": loss_g_gan.item(),
        }

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save CUT checkpoint with all models and optimizers."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'mlps_state_dict': self.mlps.state_dict(),
            'opt_g_state_dict': self.opt_g.state_dict(),
            'opt_d_state_dict': self.opt_d.state_dict(),
            'config': self.config,
            'epoch': epoch,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load CUT checkpoint.

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

        if 'mlps_state_dict' in checkpoint:
            self.mlps.load_state_dict(checkpoint['mlps_state_dict'])
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
            return (fake + 1) / 2
