"""JiT Embedding modules for patch embedding and timestep conditioning.

This module implements the core embedding components for the JiT (Just-image-Transformer)
architecture adapted for paired image-to-image translation:

Components:
- RMSNorm: Root mean square layer normalization (more stable than LayerNorm)
- BottleneckPatchEmbed: Two-stage patch embedding with compression bottleneck
- SwiGLUFFN: SwiGLU feedforward network (better gradient flow than GELU)
- TimestepEmbedder: Sinusoidal timestep embedding for flow matching

Reference:
    JiT: Towards Scaling Image Tokenizers (arXiv:2511.13720)
    https://github.com/LTH14/JiT
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes input by its RMS (root mean square) value, then scales by a
    learnable weight. More stable than LayerNorm for deep transformers as it
    does not shift the mean.

    Formula: output = x / sqrt(mean(x^2) + eps) * weight

    Args:
        dim: Dimension of the input features.
        eps: Small constant for numerical stability. Default 1e-6.

    Attributes:
        eps: Epsilon value for numerical stability.
        weight: Learnable scale parameter.

    Example:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 256, 768)
        >>> out = norm(x)
        >>> out.shape
        torch.Size([2, 256, 768])
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Dimension of the input features.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor with same shape as input.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class BottleneckPatchEmbed(nn.Module):
    """Two-stage patch embedding with bottleneck compression.

    JiT's key insight: patches should be compressed before projection to the
    transformer dimension. This prevents information overload at patch boundaries
    and improves training dynamics.

    Stage 1: Conv2d(in_chans, pca_dim, kernel_size=patch_size) - patchify + compress
    Stage 2: Conv2d(pca_dim, embed_dim, kernel_size=1) - project to transformer dim

    For 512x512 images with patch_size=32: produces 256 tokens (16x16 grid).

    Args:
        img_size: Input image size. Default 512.
        patch_size: Size of each patch. Default 32.
        in_chans: Number of input channels. Default 3.
        embed_dim: Output embedding dimension. Default 768.
        pca_dim: Intermediate compression dimension. Default 128.

    Attributes:
        num_patches: Number of output patches ((img_size // patch_size)^2).
        patch_size: Size of each patch.
        proj1: First convolution (patchify + compress).
        proj2: Second convolution (project to embed_dim).

    Example:
        >>> embed = BottleneckPatchEmbed(img_size=512, patch_size=32, embed_dim=768)
        >>> img = torch.randn(2, 3, 512, 512)
        >>> tokens = embed(img)
        >>> tokens.shape
        torch.Size([2, 256, 768])
        >>> embed.num_patches
        256
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 32,
        in_chans: int = 3,
        embed_dim: int = 768,
        pca_dim: int = 128,
    ):
        """Initialize BottleneckPatchEmbed.

        Args:
            img_size: Input image size.
            patch_size: Size of each patch.
            in_chans: Number of input channels.
            embed_dim: Output embedding dimension.
            pca_dim: Intermediate compression dimension.
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.img_size = img_size

        # Stage 1: Patchify + compress (like PCA)
        self.proj1 = nn.Conv2d(
            in_chans, pca_dim, kernel_size=patch_size, stride=patch_size
        )
        # Stage 2: Project to transformer dimension
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch tokens.

        Args:
            x: Input image tensor of shape (B, in_chans, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim).
        """
        x = self.proj1(x)  # [B, pca_dim, H/patch, W/patch]
        x = self.proj2(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class SwiGLUFFN(nn.Module):
    """SwiGLU Feedforward Network.

    SwiGLU combines Swish (SiLU) activation with gated linear units for
    improved gradient flow in deep transformers. Uses 3 linear projections
    instead of 2, with gated activation.

    Formula: output = w3(silu(w1(x)) * w2(x))

    Note: SwiGLU's 3-projection design means mlp_dim should be ~2/3 of
    typical MLP hidden size to maintain parameter count.

    Args:
        hidden_size: Input/output dimension.
        mlp_dim: Intermediate hidden dimension (after gating).

    Attributes:
        w1: Gate projection.
        w2: Value projection.
        w3: Output projection.

    Example:
        >>> ffn = SwiGLUFFN(768, 2048)
        >>> x = torch.randn(2, 256, 768)
        >>> out = ffn(x)
        >>> out.shape
        torch.Size([2, 256, 768])
    """

    def __init__(self, hidden_size: int, mlp_dim: int):
        """Initialize SwiGLUFFN.

        Args:
            hidden_size: Input/output dimension.
            mlp_dim: Intermediate hidden dimension.
        """
        super().__init__()
        self.w1 = nn.Linear(hidden_size, mlp_dim)  # Gate
        self.w2 = nn.Linear(hidden_size, mlp_dim)  # Value
        self.w3 = nn.Linear(mlp_dim, hidden_size)  # Output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feedforward transformation.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor with same shape as input.
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TimestepEmbedder(nn.Module):
    """Timestep embedding for flow matching.

    Converts scalar timesteps to high-dimensional embeddings using sinusoidal
    positional encoding followed by an MLP. This allows the model to condition
    on the diffusion/flow timestep.

    Architecture:
        1. Sinusoidal embedding: t -> [sin(t*f), cos(t*f)] for various frequencies
        2. MLP: Linear -> SiLU -> Linear

    Args:
        hidden_size: Output embedding dimension.
        frequency_embedding_size: Dimension of sinusoidal embedding. Default 256.

    Attributes:
        mlp: Two-layer MLP with SiLU activation.
        frequency_embedding_size: Size of sinusoidal embedding.

    Example:
        >>> t_emb = TimestepEmbedder(768)
        >>> t = torch.rand(2)  # Timesteps in [0, 1]
        >>> out = t_emb(t)
        >>> out.shape
        torch.Size([2, 768])
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """Initialize TimestepEmbedder.

        Args:
            hidden_size: Output embedding dimension.
            frequency_embedding_size: Dimension of sinusoidal embedding.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def sinusoidal_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Compute sinusoidal positional embedding for timesteps.

        Creates embeddings using sine and cosine functions at various
        frequencies, similar to transformer positional encodings.

        Args:
            t: Timestep tensor of shape (B,).
            dim: Dimension of the output embedding.
            max_period: Maximum period for the frequencies.

        Returns:
            Sinusoidal embedding of shape (B, dim).
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep tensor of shape (B,) with values typically in [0, 1].

        Returns:
            Timestep embedding of shape (B, hidden_size).
        """
        t_emb = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_emb)
