"""JiT (Just-image-Transformer) for conditional image-to-image translation.

This module provides a complete JiT-based diffusion transformer for paired I2I
translation with flow matching, adapted for virtual multiplex IHC generation:

High-level Models:
- ConditionalJiT: Complete model for paired I2I translation
- SourceEncoder: V3 ConvNeXt wrapper for source conditioning

Building Blocks:
- Embeddings: RMSNorm, BottleneckPatchEmbed, SwiGLUFFN, TimestepEmbedder
- Attention: Self-attention and CrossAttention with QK-Norm + Flash Attention 2
- Blocks: ConditionalJiTBlock with AdaLN-Zero modulation

Example:
    >>> from multiplex.models.jit import ConditionalJiT
    >>> model = ConditionalJiT(depth=12, hidden_size=768)
    >>> noisy = torch.randn(1, 3, 512, 512)
    >>> t = torch.tensor([0.5])
    >>> source = torch.randn(1, 4, 512, 512)
    >>> avail = torch.tensor([1])
    >>> output = model(noisy, t, source, avail)  # (1, 3, 512, 512)

Reference:
    JiT: Towards Scaling Image Tokenizers (arXiv:2511.13720)
    https://github.com/LTH14/JiT
"""

from multiplex.models.jit.attention import Attention, CrossAttention
from multiplex.models.jit.blocks import ConditionalJiTBlock
from multiplex.models.jit.conditional_jit import ConditionalJiT
from multiplex.models.jit.embeddings import (
    BottleneckPatchEmbed,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
)
from multiplex.models.jit.source_encoder import SourceEncoder
from multiplex.models.marker_gnn import MarkerGNN

__all__ = [
    # High-level models
    "ConditionalJiT",
    "SourceEncoder",
    # Graph modules (Phase 23)
    "MarkerGNN",
    # Embeddings
    "RMSNorm",
    "BottleneckPatchEmbed",
    "SwiGLUFFN",
    "TimestepEmbedder",
    # Attention
    "Attention",
    "CrossAttention",
    # Blocks
    "ConditionalJiTBlock",
]
