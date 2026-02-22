"""JiT Attention modules with QK-Norm and Flash Attention 2.

This module implements self-attention and cross-attention for the JiT transformer:

Components:
- Attention: Self-attention with optional QK-Norm to prevent entropy collapse
- CrossAttention: Cross-attention for conditioning on source encoder tokens

Both use F.scaled_dot_product_attention (SDPA) which automatically uses
Flash Attention 2 on compatible hardware (PyTorch >= 2.2.0).

QK-Norm: Normalizing Q and K before the dot product prevents attention entropy
collapse in deep transformers, where attention weights become nearly one-hot.

Reference:
    JiT: Towards Scaling Image Tokenizers (arXiv:2511.13720)
    https://github.com/LTH14/JiT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    """Self-attention with optional QK-Norm.

    Standard multi-head self-attention with:
    - Combined QKV projection for efficiency
    - Optional QK-Norm to prevent entropy collapse in deep networks
    - F.scaled_dot_product_attention for Flash Attention 2 support

    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply L2 normalization to Q and K. Default True.

    Attributes:
        hidden_size: Input/output dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (hidden_size // num_heads).
        qkv: Combined QKV projection.
        proj: Output projection.
        q_norm: LayerNorm for Q if qk_norm=True.
        k_norm: LayerNorm for K if qk_norm=True.

    Example:
        >>> attn = Attention(768, 12)
        >>> x = torch.randn(2, 256, 768)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 256, 768])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
    ):
        """Initialize Attention.

        Args:
            hidden_size: Input/output dimension.
            num_heads: Number of attention heads.
            qk_norm: Whether to apply normalization to Q and K.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Combined QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK-Norm to prevent attention entropy collapse
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor of shape (B, N, D) where:
                B = batch size
                N = sequence length (number of tokens)
                D = hidden_size

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK-Norm if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention (uses Flash Attention 2 when available)
        out = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project output
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)

        return out


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on source tokens.

    Implements cross-attention where:
    - Q comes from target tokens (decoder)
    - K, V come from source tokens (encoder)

    This is the key mechanism for image-to-image translation, allowing
    target patches to attend to relevant source patches.

    Args:
        hidden_size: Input/output dimension.
        num_heads: Number of attention heads.
        qk_norm: Whether to apply L2 normalization to Q and K. Default True.

    Attributes:
        hidden_size: Input/output dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (hidden_size // num_heads).
        to_q: Query projection.
        to_kv: Combined Key-Value projection.
        proj: Output projection.
        q_norm: LayerNorm for Q if qk_norm=True.
        k_norm: LayerNorm for K if qk_norm=True.

    Example:
        >>> cross = CrossAttention(768, 12)
        >>> q = torch.randn(2, 256, 768)  # target tokens
        >>> kv = torch.randn(2, 256, 768)  # source tokens
        >>> out = cross(q, kv)
        >>> out.shape
        torch.Size([2, 256, 768])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
    ):
        """Initialize CrossAttention.

        Args:
            hidden_size: Input/output dimension.
            num_heads: Number of attention heads.
            qk_norm: Whether to apply normalization to Q and K.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Separate projections for cross-attention
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_kv = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK-Norm to prevent attention entropy collapse
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            x: Query tensor (target tokens) of shape (B, N, D).
            context: Key-Value tensor (source tokens) of shape (B, M, D).
                N and M can differ (different sequence lengths).

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        # Compute Q from target, K/V from source
        q = self.to_q(x)
        kv = self.to_kv(context)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        kv = rearrange(kv, "b m (two h d) -> two b h m d", two=2, h=self.num_heads)
        k, v = kv[0], kv[1]

        # Apply QK-Norm if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention (uses Flash Attention 2 when available)
        out = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project output
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)

        return out
