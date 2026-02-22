"""JiT Transformer blocks with AdaLN-Zero modulation.

This module implements the conditional transformer block for JiT-based
image-to-image translation:

Components:
- modulate: AdaLN modulation helper function
- ConditionalJiTBlock: Full block with self-attn + cross-attn + FFN

The block uses AdaLN-Zero modulation where the conditioning signal (timestep +
availability embedding) produces shift, scale, and gate parameters for each
sublayer. This allows fine-grained conditioning control.

Reference:
    JiT: Towards Scaling Image Tokenizers (arXiv:2511.13720)
    DiT: Scalable Diffusion Models with Transformers (AdaLN-Zero)
"""

import torch
import torch.nn as nn

from multiplex.models.jit.attention import Attention, CrossAttention
from multiplex.models.jit.embeddings import RMSNorm, SwiGLUFFN


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation to input tensor.

    Formula: y = x * (1 + scale) + shift

    The shift and scale tensors have shape (B, D) and are broadcast
    over the sequence dimension.

    Args:
        x: Input tensor of shape (B, N, D).
        shift: Shift parameter of shape (B, D).
        scale: Scale parameter of shape (B, D).

    Returns:
        Modulated tensor of shape (B, N, D).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ConditionalJiTBlock(nn.Module):
    """Conditional transformer block with AdaLN-Zero modulation.

    A full transformer block consisting of:
    1. Self-attention on target tokens (with AdaLN modulation)
    2. Cross-attention from target to source tokens (with AdaLN modulation)
    3. SwiGLU feedforward network (with AdaLN modulation)

    Each sublayer has 3 modulation parameters (shift, scale, gate), for a
    total of 9 parameters produced by a single AdaLN MLP.

    The gating mechanism (from AdaLN-Zero) allows the conditioning to
    completely suppress sublayer outputs, which is useful for learning
    to ignore certain sublayers early in training.

    Args:
        hidden_size: Dimension of the transformer.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for FFN. Default 4.0.
            Note: SwiGLU uses ~2/3 of this due to 3-projection design.
        qk_norm: Whether to use QK-Norm in attention. Default True.

    Attributes:
        norm1: RMSNorm before self-attention.
        self_attn: Self-attention module.
        norm_cross: RMSNorm before cross-attention.
        cross_attn: Cross-attention module.
        norm2: RMSNorm before FFN.
        mlp: SwiGLU feedforward network.
        adaLN_modulation: MLP producing 9 modulation parameters.

    Example:
        >>> block = ConditionalJiTBlock(hidden_size=768, num_heads=12)
        >>> x = torch.randn(2, 256, 768)        # target tokens
        >>> source = torch.randn(2, 256, 768)   # source encoder tokens
        >>> c = torch.randn(2, 768)             # conditioning (time + availability)
        >>> out = block(x, source, c)
        >>> out.shape
        torch.Size([2, 256, 768])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
    ):
        """Initialize ConditionalJiTBlock.

        Args:
            hidden_size: Dimension of the transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for FFN.
            qk_norm: Whether to use QK-Norm in attention.
        """
        super().__init__()

        # Self-attention
        self.norm1 = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, qk_norm=qk_norm)

        # Cross-attention (target queries source)
        self.norm_cross = RMSNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads, qk_norm=qk_norm)

        # Feed-forward network
        # SwiGLU uses ~2/3 of typical MLP hidden size due to 3-projection design
        mlp_hidden = int(hidden_size * mlp_ratio * 2 / 3)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden)

        # AdaLN-Zero modulation: produces 9 parameters
        # (shift, scale, gate) x 3 for (self-attn, cross-attn, ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        source_tokens: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Apply conditional transformer block.

        Args:
            x: Target tokens of shape (B, N, D).
            source_tokens: Source encoder output of shape (B, M, D).
            c: Conditioning embedding of shape (B, D).
                Combined timestep + availability embedding.

        Returns:
            Output tokens of shape (B, N, D).
        """
        # Compute all 9 modulation parameters at once
        mods = self.adaLN_modulation(c).chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = mods[0], mods[1], mods[2]
        shift_ca, scale_ca, gate_ca = mods[3], mods[4], mods[5]
        shift_ff, scale_ff, gate_ff = mods[6], mods[7], mods[8]

        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(x_norm)

        # Cross-attention with modulation
        x_norm = modulate(self.norm_cross(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(x_norm, source_tokens)

        # FFN with modulation
        x_norm = modulate(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff.unsqueeze(1) * self.mlp(x_norm)

        return x
