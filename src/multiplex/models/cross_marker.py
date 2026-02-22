"""Cross-marker attention modules for inter-marker relationship modeling.

This module implements a two-stage cross-marker attention architecture that
enables each marker prediction to leverage information from all other markers,
improving biological consistency in multi-marker virtual staining.

Stage 1 (CrossMarkerAttention):
    Operates at the encoder bottleneck (1/32 resolution) to model inter-marker
    dependencies in feature space. Uses multi-head self-attention with gated
    fusion for stable integration.

Stage 2 (OutputRefinementModule):
    Applies cross-channel attention on output predictions, allowing each
    marker prediction to refine based on all others. Uses learned residual
    weighting for controlled contribution.

Both stages store attention maps for visualization and interpretability,
and can be individually disabled for ablation studies.

Reference:
    CNN-Transformer Gated Fusion Network (Scientific Reports, May 2025)
    Marker Sampling and Excite (Nature Machine Intelligence, 2021)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CrossMarkerConfig:
    """Configuration for cross-marker attention modules.

    Controls both Stage 1 (feature-level attention at bottleneck) and Stage 2
    (output refinement) behavior. Individual stages can be disabled for
    ablation studies.

    Attributes:
        use_stage1: Enable Stage 1 feature-level attention. Default True.
        stage1_embed_dim: Embedding dimension for Stage 1 MHA, must match
            encoder bottleneck channels. Default 1024.
        stage1_num_heads: Number of attention heads for Stage 1. Default 8.
        stage1_dropout: Dropout probability for Stage 1 attention. Default 0.1.
        use_stage2: Enable Stage 2 output refinement. Default True.
        stage2_hidden_dim: Hidden dimension for Stage 2 projections. Default 64.
        stage2_num_heads: Number of attention heads for Stage 2. Default 4.
        stage2_dropout: Dropout probability for Stage 2 attention. Default 0.1.
        stage2_bypass: Skip Stage 2 at inference for faster prediction. Default False.
        num_markers: Number of output markers. Default 5.

    Example:
        >>> config = CrossMarkerConfig()
        >>> config.use_stage1
        True
        >>> config.stage1_embed_dim
        1024

        >>> # Disable Stage 2 for ablation
        >>> config_no_s2 = CrossMarkerConfig(use_stage2=False)
    """

    use_stage1: bool = True
    stage1_embed_dim: int = 1024
    stage1_num_heads: int = 8
    stage1_dropout: float = 0.1
    use_stage2: bool = True
    stage2_hidden_dim: int = 64
    stage2_num_heads: int = 4
    stage2_dropout: float = 0.1
    stage2_bypass: bool = False
    num_markers: int = 5


class CrossMarkerAttention(nn.Module):
    """Stage 1: Cross-marker attention at encoder bottleneck.

    Applies multi-head self-attention so each spatial position's features
    can attend to all other positions, learning global marker relationships.
    Uses gated fusion for stable integration: output = original + gate * (attended - original).

    Operates at 1/32 resolution (16x16 spatial for 512x512 input).

    The gate is a 2-layer MLP with GELU activation that learns when attention
    should contribute vs when to preserve original features.

    Args:
        config: CrossMarkerConfig with Stage 1 parameters.

    Attributes:
        mha: Multi-head attention module.
        gate_proj: 2-layer MLP for gated fusion.
        norm_pre: LayerNorm applied before attention.
        norm_post: LayerNorm applied after gated fusion.
        _last_attention: Stored attention weights from last forward pass.

    Example:
        >>> config = CrossMarkerConfig()
        >>> cma = CrossMarkerAttention(config)
        >>> features = torch.randn(2, 1024, 16, 16)  # Bottleneck features
        >>> attended = cma(features)
        >>> attended.shape
        torch.Size([2, 1024, 16, 16])

        >>> # Access attention maps for visualization
        >>> attn_map = cma.get_attention_map()
        >>> attn_map.shape  # (B, num_heads, H*W, H*W)
        torch.Size([2, 8, 256, 256])
    """

    def __init__(self, config: CrossMarkerConfig):
        """Initialize CrossMarkerAttention.

        Args:
            config: CrossMarkerConfig with Stage 1 parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.stage1_embed_dim
        self.num_heads = config.stage1_num_heads

        # Multi-head attention with batch_first=True for (B, seq, features) format
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=config.stage1_dropout,
            batch_first=True,
        )

        # Gated fusion: 2-layer MLP with GELU
        # output = original + sigmoid(gate) * (attended - original)
        self.gate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Layer normalization for stability
        self.norm_pre = nn.LayerNorm(self.embed_dim)
        self.norm_post = nn.LayerNorm(self.embed_dim)

        # Store attention weights for visualization (detached)
        self._last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-marker attention with gated fusion.

        All spatial positions attend to all other positions,
        learning global marker relationships.

        Args:
            x: Bottleneck features (B, C, H, W) where C = stage1_embed_dim.

        Returns:
            Attended features with same shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # Reshape to sequence: (B, H*W, C)
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

        # Gated fusion: output = original + sigmoid(gate) * (attended - original)
        gate_input = torch.cat([x_flat, attended], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        fused = x_flat + gate * (attended - x_flat)

        # Post-norm and reshape back to spatial
        fused = self.norm_post(fused)
        return fused.transpose(1, 2).view(B, C, H, W)

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass.

        Returns:
            Attention weights (B, num_heads, H*W, H*W) or None if forward
            hasn't been called yet.
        """
        return self._last_attention


class OutputRefinementModule(nn.Module):
    """Stage 2: Cross-channel attention on output predictions.

    Each marker prediction attends to all other marker predictions,
    allowing refinement based on inter-marker consistency.

    Uses learned residual: output = pred + sigmoid(alpha) * (refined - pred)
    where alpha starts at 0 (sigmoid(0) = 0.5, giving balanced contribution).

    To handle memory constraints, attention is computed at a reduced resolution
    (32x32 by default) and the refinement is upsampled back to full resolution.
    This makes the attention matrix tractable (32x32 = 1024 tokens vs 512x512 = 262K).

    Args:
        config: CrossMarkerConfig with Stage 2 parameters.

    Attributes:
        proj_in: 1x1 conv projecting num_markers to hidden_dim.
        mha: Multi-head attention module.
        proj_out: 1x1 conv projecting hidden_dim to num_markers.
        alpha: Learned residual weight parameter, initialized to 0.
        norm: LayerNorm for attended features.
        _bypass: Flag to skip refinement for fast inference.
        _last_attention: Stored attention weights from last forward pass.
        _attn_resolution: Target resolution for attention computation.

    Example:
        >>> config = CrossMarkerConfig()
        >>> orm = OutputRefinementModule(config)
        >>> pred = torch.randn(2, 5, 512, 512)
        >>> refined = orm(pred)
        >>> refined.shape
        torch.Size([2, 5, 512, 512])

        >>> # Check learned alpha weight
        >>> orm.get_alpha()  # Returns value in [0, 1]
        0.5

        >>> # Enable bypass for faster inference
        >>> orm.set_bypass(True)
        >>> orm(pred) is pred  # No refinement applied
        True
    """

    # Target resolution for attention to keep memory tractable
    # 32x32 = 1024 tokens, attention matrix is 1024x1024 = 1M elements per head
    _attn_resolution: int = 32

    def __init__(self, config: CrossMarkerConfig):
        """Initialize OutputRefinementModule.

        Args:
            config: CrossMarkerConfig with Stage 2 parameters.
        """
        super().__init__()
        self.config = config
        self.num_markers = config.num_markers
        self.hidden_dim = config.stage2_hidden_dim

        # Project markers to hidden dim for attention
        self.proj_in = nn.Conv2d(
            config.num_markers, self.hidden_dim, kernel_size=1
        )

        # Cross-channel attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.stage2_num_heads,
            dropout=config.stage2_dropout,
            batch_first=True,
        )

        # Project back to markers
        self.proj_out = nn.Conv2d(
            self.hidden_dim, config.num_markers, kernel_size=1
        )

        # Learned residual weight, initialized to 0 (sigmoid(0) = 0.5)
        # output = pred + sigmoid(alpha) * (refined - pred)
        self.alpha = nn.Parameter(torch.zeros(1))

        # LayerNorm for attended features
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Bypass flag for fast inference
        self._bypass = config.stage2_bypass

        # Store attention for visualization
        self._last_attention: Optional[torch.Tensor] = None

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """Refine predictions via cross-channel attention.

        Attention is computed at reduced resolution (32x32) for memory efficiency,
        then the refinement is upsampled and blended with the full-resolution input.

        Args:
            pred: Initial predictions (B, num_markers, H, W).

        Returns:
            Refined predictions (B, num_markers, H, W).
        """
        if self._bypass:
            return pred

        B, M, H, W = pred.shape

        # Downsample for memory-efficient attention
        target_h = min(H, self._attn_resolution)
        target_w = min(W, self._attn_resolution)
        needs_resize = (H > target_h) or (W > target_w)

        if needs_resize:
            pred_small = nn.functional.interpolate(
                pred, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        else:
            pred_small = pred

        # Project to hidden representation
        x = self.proj_in(pred_small)  # (B, hidden_dim, target_h, target_w)

        # Reshape for attention: (B, target_h*target_w, hidden_dim)
        x_flat = x.flatten(2).transpose(1, 2)

        # Self-attention across all spatial positions
        attended, attn_weights = self.mha(
            x_flat, x_flat, x_flat,
            need_weights=True,
            average_attn_weights=False,
        )

        # Store for visualization (detached)
        self._last_attention = attn_weights.detach()

        # Normalize and project back
        attended = self.norm(attended)
        attended = attended.transpose(1, 2).view(B, self.hidden_dim, target_h, target_w)
        refined_small = self.proj_out(attended)

        # Upsample refined output back to original resolution
        if needs_resize:
            refined = nn.functional.interpolate(
                refined_small, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            refined = refined_small

        # Learned residual: pred + sigmoid(alpha) * (refined - pred)
        alpha = torch.sigmoid(self.alpha)
        return pred + alpha * (refined - pred)

    def set_bypass(self, bypass: bool = True) -> None:
        """Enable/disable bypass for inference.

        When bypass is enabled, forward() returns input unchanged
        for faster inference.

        Args:
            bypass: Whether to skip refinement. Default True.
        """
        self._bypass = bypass

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass.

        Returns:
            Attention weights (B, num_heads, H*W, H*W) or None if forward
            hasn't been called yet or bypass is enabled.
        """
        return self._last_attention

    def get_alpha(self) -> float:
        """Get current learned residual weight.

        Returns:
            Sigmoid of alpha parameter, in range [0, 1].
            Value of 0.5 means balanced contribution (alpha=0).
            Value closer to 0 means less refinement contribution.
            Value closer to 1 means more refinement contribution.
        """
        return torch.sigmoid(self.alpha).item()
