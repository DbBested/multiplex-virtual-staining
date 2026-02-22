"""PatchGAN discriminator for adversarial training.

This module implements a 70x70 PatchGAN discriminator that operates on
concatenated brightfield + marker images for adversarial training in
the virtual staining task.

The discriminator outputs patch-level real/fake predictions, which allows
it to penalize structure at the scale of image patches rather than globally.

Architecture:
    Input: (B, 6, H, W) - concatenated BF (1) + markers (5)
    Layers: 5 conv layers with 4x4 kernels, strides [2,2,2,1,1]
    Output: (B, 1, H', W') - patch-level real/fake scores

Reference:
    Architecture based on pix2pix PatchGAN (Isola et al., 2017).
    70x70 receptive field determined by kernel and stride configuration.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm


class PatchGAN70(nn.Module):
    """70x70 PatchGAN discriminator with spectral normalization.

    Fully convolutional discriminator that outputs patch-level real/fake
    predictions. The 70x70 receptive field allows discrimination of local
    texture and structure.

    For 512x512 input, output is approximately 30x30 patches.

    Attributes:
        model: Sequential network of conv layers.
        use_spectral_norm: Whether spectral normalization is applied.

    Example:
        >>> disc = PatchGAN70(in_channels=6, ndf=64)
        >>> bf = torch.randn(1, 1, 512, 512)
        >>> markers = torch.randn(1, 5, 512, 512)
        >>> out = disc(bf, markers)
        >>> out.shape
        torch.Size([1, 1, 30, 30])
    """

    def __init__(
        self,
        in_channels: int = 6,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        """Initialize PatchGAN70 discriminator.

        Args:
            in_channels: Number of input channels. Default 6 (BF=1 + markers=5).
            ndf: Base number of discriminator filters. Default 64.
            n_layers: Number of middle convolutional layers. Default 3 gives
                70x70 receptive field with proper stride configuration.
            use_spectral_norm: Apply spectral normalization to Conv2d layers
                for training stability. Default True.
        """
        super().__init__()

        self.use_spectral_norm = use_spectral_norm

        # Build discriminator layers
        # Layer configuration for 70x70 receptive field:
        # Layer 1: Conv(in, ndf, 4, 2, 1)    -> RF: 4
        # Layer 2: Conv(ndf, ndf*2, 4, 2, 1) -> RF: 10
        # Layer 3: Conv(ndf*2, ndf*4, 4, 2, 1) -> RF: 22
        # Layer 4: Conv(ndf*4, ndf*8, 4, 1, 1) -> RF: 46
        # Layer 5: Conv(ndf*8, 1, 4, 1, 1)     -> RF: 70

        layers = []

        # Layer 1: No normalization, LeakyReLU
        conv1 = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=True)
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
        layers.extend([
            conv1,
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Middle layers with InstanceNorm
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            conv = nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,  # No bias when using normalization
            )
            if use_spectral_norm:
                conv = spectral_norm(conv)

            layers.extend([
                conv,
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Layer with stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        conv_stride1 = nn.Conv2d(
            ndf * nf_mult_prev,
            ndf * nf_mult,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=False,
        )
        if use_spectral_norm:
            conv_stride1 = spectral_norm(conv_stride1)

        layers.extend([
            conv_stride1,
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Final output layer: 1 channel output
        conv_final = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            conv_final = spectral_norm(conv_final)
        layers.append(conv_final)

        self.model = nn.Sequential(*layers)

    def forward(self, bf: Tensor, markers: Tensor) -> Tensor:
        """Forward pass: concatenate BF and markers, produce patch predictions.

        Args:
            bf: Brightfield input, shape (B, 1, H, W).
            markers: Marker predictions or ground truth, shape (B, num_markers, H, W).
                For default setup, num_markers=5.

        Returns:
            Patch-level real/fake scores, shape (B, 1, H', W').
            For 512x512 input with default settings: (B, 1, 30, 30).
        """
        # Concatenate along channel dimension: BF first, then markers
        x = torch.cat([bf, markers], dim=1)
        return self.model(x)

    def get_receptive_field(self) -> int:
        """Return the theoretical receptive field size.

        Returns:
            Receptive field size in pixels (70 for default configuration).
        """
        # 70x70 receptive field for n_layers=3
        return 70

    def get_parameter_count(self) -> int:
        """Get total number of parameters.

        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())


def calculate_receptive_field(kernel_sizes: list, strides: list) -> int:
    """Calculate receptive field for a sequence of conv layers.

    Args:
        kernel_sizes: List of kernel sizes for each layer.
        strides: List of strides for each layer.

    Returns:
        Total receptive field size.

    Example:
        >>> # Standard PatchGAN70 configuration
        >>> ks = [4, 4, 4, 4, 4]
        >>> strides = [2, 2, 2, 1, 1]
        >>> calculate_receptive_field(ks, strides)
        70
    """
    rf = 1
    accumulated_stride = 1

    for k, s in zip(kernel_sizes, strides):
        rf = rf + (k - 1) * accumulated_stride
        accumulated_stride *= s

    return rf
