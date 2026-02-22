"""Progressive CNN decoder for JiT unpatchify.

Replaces the single linear unpatchify layer with a multi-stage CNN decoder
that progressively upsamples from the transformer token grid to full image
resolution. This allows learning of fine spatial patterns rather than
forcing a single linear layer to map 768-dim → 32*32*3 pixel values.

Architecture:
    Tokens (B, N, D) → reshape to (B, D, h, w) → progressive upsample stages
    Each stage: Conv → GroupNorm → GELU → PixelShuffle(2x)
    Final stage: Conv → output channels (no activation)

For img_size=512:
    patch_size=32 → 16x16 grid → 5 upsample stages → 512x512
    patch_size=16 → 32x32 grid → 4 upsample stages → 512x512

Reference:
    DeCo: Frequency-Decoupled Pixel Diffusion (arXiv:2511.19365)
    PixelDiT: Pixel Diffusion Transformers (arXiv:2511.20645)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUpsampleStage(nn.Module):
    """Single 2x upsample stage using bilinear interpolation + convolution.

    bilinear 2x upsample → Conv2d(3x3) → GroupNorm → GELU

    Uses bilinear upsampling instead of PixelShuffle to naturally share
    spatial information across patch boundaries without rearrangement artifacts.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # Find largest valid group count: prefer 32, fallback to divisor
        num_groups = min(32, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


class ConvUpsampleDecoder(nn.Module):
    """Bilinear+conv decoder replacing linear unpatchify.

    Takes transformer tokens at a spatial grid resolution and progressively
    upsamples to full image resolution. Each stage uses bilinear 2x upsampling
    followed by Conv2d + GroupNorm + GELU, which naturally shares spatial
    information across patch boundaries (unlike independent per-patch linear
    projection or PixelShuffle rearrangement).

    The final output conv is zero-initialized so the decoder starts as
    near-identity, compatible with residual training.

    Args:
        hidden_size: Transformer hidden dimension (token dim). Default 768.
        out_chans: Number of output image channels. Default 3.
        img_size: Target image spatial size. Default 512.
        patch_size: Patch size used in the transformer. Default 32.

    Example:
        >>> decoder = ConvUpsampleDecoder(768, 3, 512, 16)
        >>> tokens = torch.randn(2, 1024, 768)  # 32x32 grid
        >>> img = decoder(tokens)
        >>> img.shape
        torch.Size([2, 3, 512, 512])
    """

    def __init__(
        self,
        hidden_size: int = 768,
        out_chans: int = 3,
        img_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.hidden_size = hidden_size

        num_stages = int(math.log2(patch_size))

        # Channel schedule: halve channels each stage, minimum 48
        channels = [hidden_size]
        ch = hidden_size
        for _ in range(num_stages):
            ch = max(ch // 2, 48)
            channels.append(ch)

        # Upsample stages: bilinear 2x + Conv + GroupNorm + GELU
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(ConvUpsampleStage(channels[i], channels[i + 1]))

        # Final output projection (zero-initialized for residual-friendly start)
        self.output_proj = nn.Conv2d(channels[-1], out_chans, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init final output conv for residual-friendly start."""
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode tokens to image.

        Args:
            x: Transformer tokens of shape (B, N, D).

        Returns:
            Image of shape (B, out_chans, img_size, img_size).
        """
        B, N, D = x.shape
        h = w = self.grid_size
        # Reshape to spatial grid
        x = x.transpose(1, 2).reshape(B, D, h, w)

        for stage in self.stages:
            x = stage(x)
        return self.output_proj(x)


class ResConvBlock(nn.Module):
    """Residual convolution block with GroupNorm.

    Conv → GroupNorm → GELU → Conv → GroupNorm → residual add → GELU

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class UpsampleStage(nn.Module):
    """Single 2x upsample stage using PixelShuffle.

    ResConvBlock → Conv(ch → out_ch*4) → PixelShuffle(2) → (out_ch, 2H, 2W)

    Uses PixelShuffle (sub-pixel convolution) instead of ConvTranspose2d
    to avoid checkerboard artifacts.

    Args:
        in_ch: Input channels.
        out_ch: Output channels (after PixelShuffle).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res_block = ResConvBlock(in_ch)
        # PixelShuffle: needs 4x output channels before shuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        return self.upsample(x)


class ProgressiveDecoder(nn.Module):
    """Progressive CNN decoder replacing linear unpatchify.

    Takes transformer tokens at a spatial grid resolution and progressively
    upsamples to full image resolution through multiple stages.

    Args:
        hidden_size: Transformer hidden dimension (token dim). Default 768.
        out_chans: Number of output image channels. Default 3.
        img_size: Target image spatial size. Default 512.
        patch_size: Patch size used in the transformer. Default 32.

    Example:
        >>> decoder = ProgressiveDecoder(768, 3, 512, 32)
        >>> tokens = torch.randn(2, 256, 768)  # 16x16 grid
        >>> img = decoder(tokens)
        >>> img.shape
        torch.Size([2, 3, 512, 512])
    """

    def __init__(
        self,
        hidden_size: int = 768,
        out_chans: int = 3,
        img_size: int = 512,
        patch_size: int = 32,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.hidden_size = hidden_size

        num_stages = int(math.log2(patch_size))
        # Channel schedule: halve channels each stage, minimum 64
        channels = [hidden_size]
        ch = hidden_size
        for _ in range(num_stages - 1):
            ch = max(ch // 2, 64)
            channels.append(ch)
        # Last upsample stage outputs an intermediate dim
        channels.append(max(ch // 2, 32))

        # Initial projection to reduce dimensionality before upsampling
        self.input_proj = nn.Sequential(
            nn.Conv2d(hidden_size, channels[0], 1),
            nn.GroupNorm(min(32, channels[0]), channels[0]),
            nn.GELU(),
        )

        # Upsample stages
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(UpsampleStage(channels[i], channels[i + 1]))

        # Final output projection (no activation - let training determine range)
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.GroupNorm(min(32, channels[-1]), channels[-1]),
            nn.GELU(),
            nn.Conv2d(channels[-1], out_chans, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize final output conv to small values for residual-friendly start."""
        # Small init on the final conv so initial output is near zero
        # (compatible with both x_prediction and velocity modes)
        nn.init.normal_(self.output_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode tokens to image.

        Args:
            x: Transformer tokens of shape (B, N, D).

        Returns:
            Image of shape (B, out_chans, img_size, img_size).
        """
        B, N, D = x.shape
        h = w = self.grid_size
        # Reshape to spatial grid
        x = x.transpose(1, 2).reshape(B, D, h, w)

        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x)
        return self.output_proj(x)
