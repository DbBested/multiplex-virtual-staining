"""Cross-stain attention module for input fusion.

This module implements self-attention for fusing features from multiple input stains
(IHC, Hematoxylin) into a unified representation. Unlike CrossMarkerAttention which
operates on OUTPUT predictions to enforce consistency, CrossStainAttention operates
on INPUTS before encoding to learn which spatial regions are most informative.

The key innovation is that cross-stain attention operates at the input projection
level (after MultiStainInputProjection), allowing the model to learn which spatial
regions of each input stain contribute most to the translation task.

Architecture:
    MultiStainInputProjection -> CrossStainAttention -> Encoder -> ...
                                     ^
                                     |
                           (This module)

Reference:
    Cross-Stain Contrastive Learning for Multiplex Stain Analysis
    PyTorch nn.MultiheadAttention for efficient attention computation
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CrossStainAttentionConfig:
    """Configuration for cross-stain attention module.

    Controls the self-attention mechanism that fuses input stain features.
    The embed_dim must match the output of MultiStainInputProjection (default 64).

    Attributes:
        embed_dim: Embedding dimension for attention. Must match
            MultiStainInputProjection output. Default 64.
        num_heads: Number of attention heads. embed_dim must be divisible
            by num_heads. Default 4.
        dropout: Dropout probability for attention weights. Default 0.1.
        use_gated_fusion: Whether to use gated fusion (sigmoid gate) or
            simple residual addition. Gated fusion provides more stable
            training and controlled contribution. Default True.

    Example:
        >>> config = CrossStainAttentionConfig()
        >>> config.embed_dim
        64
        >>> config.num_heads
        4

        >>> # Custom configuration
        >>> config = CrossStainAttentionConfig(embed_dim=128, num_heads=8)
    """

    embed_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    use_gated_fusion: bool = True


class CrossStainAttention(nn.Module):
    """Cross-stain self-attention for input fusion.

    Operates after MultiStainInputProjection, before encoder. All spatial
    positions attend to all other positions via self-attention, learning
    which regions of the fused input are most informative for translation.

    Unlike CrossMarkerAttention (output consistency), this fuses INPUTS.

    The attention mechanism allows the model to:
    - Learn spatial relationships between input stain features
    - Weight regions based on their informativeness
    - Fuse multi-stain information coherently

    Args:
        config: CrossStainAttentionConfig with attention parameters.

    Attributes:
        config: Stored configuration.
        embed_dim: Embedding dimension from config.
        num_heads: Number of attention heads from config.
        mha: Multi-head self-attention module.
        gate_proj: Gated fusion MLP (if use_gated_fusion=True).
        norm_pre: LayerNorm applied before attention.
        norm_post: LayerNorm applied after fusion.
        _last_attention: Stored attention weights from last forward pass.

    Example:
        >>> config = CrossStainAttentionConfig()
        >>> csa = CrossStainAttention(config)
        >>> x = torch.randn(2, 64, 128, 128)  # Projected input
        >>> fused = csa(x)
        >>> fused.shape
        torch.Size([2, 64, 128, 128])

        >>> # Access attention maps for visualization
        >>> attn_map = csa.get_attention_map()
        >>> attn_map.shape  # (B, num_heads, H*W, H*W)
        torch.Size([2, 4, 16384, 16384])
    """

    def __init__(self, config: CrossStainAttentionConfig):
        """Initialize CrossStainAttention.

        Args:
            config: CrossStainAttentionConfig with attention parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads

        # Multi-head self-attention with batch_first=True for (B, seq, features) format
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Gated fusion: output = orig + sigmoid(gate) * (attended - orig)
        # 2-layer MLP with GELU activation
        if config.use_gated_fusion:
            self.gate_proj = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.gate_proj = None

        # Layer normalization for stability
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.norm_post = nn.LayerNorm(self.embed_dim)

        # Store attention weights for visualization (detached)
        self._last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        avail_config: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse input features via self-attention.

        All spatial positions attend to all other positions, learning
        which regions are most informative for the translation task.

        Args:
            x: Projected features from MultiStainInputProjection.
                Shape: (B, embed_dim, H, W)
            avail_config: Optional availability configuration indices.
                Shape: (B,) with values in {0, 1, 2}.
                Currently passed through for future attention masking
                based on input availability (not used in v1).

        Returns:
            Fused features with same shape (B, embed_dim, H, W).
            When use_gated_fusion=True:
                output = original + sigmoid(gate) * (attended - original)
            When use_gated_fusion=False:
                output = original + attended

        Note:
            avail_config is accepted but not currently used. It is included
            for future extension where attention could be masked based on
            which input modalities are actually available.
        """
        B, C, H, W = x.shape

        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm_pre(x_flat)

        # Self-attention (all positions attend to all)
        attended, attn_weights = self.mha(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=True,
            average_attn_weights=False,  # Keep per-head weights
        )

        # Store for visualization (detached to avoid memory growth)
        self._last_attention = attn_weights.detach()

        # Gated fusion or simple residual
        if self.gate_proj is not None:
            # Gated fusion: output = orig + sigmoid(gate) * (attended - orig)
            gate_input = torch.cat([x_flat, attended], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            fused = x_flat + gate * (attended - x_flat)
        else:
            # Simple residual addition
            fused = x_flat + attended

        # Post-norm and reshape back to spatial
        fused = self.norm_post(fused)
        return fused.transpose(1, 2).view(B, C, H, W)

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass.

        The attention map shows how each spatial position attends to
        all other positions, useful for visualizing which regions
        the model finds most informative.

        Returns:
            Attention weights tensor of shape (B, num_heads, H*W, H*W)
            or None if forward hasn't been called yet.

        Note:
            The returned tensor is detached from the computation graph
            to prevent memory accumulation during training.
        """
        return self._last_attention
