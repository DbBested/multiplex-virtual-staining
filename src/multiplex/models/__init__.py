"""Model modules for Multiplex Virtual IHC.

This package contains the neural network architectures for brightfield to
multi-marker prediction, including:

Core Modules:
- ConvNeXtEncoder: Pretrained ConvNeXt-Base encoder with multi-scale features
- AttentionGate: Attention mechanism for skip connection filtering
- DecoderBlock: Decoder block with attention-gated skip connections
- MultiHeadOutput: Multiple output heads for marker-specific predictions

Models:
- AttentionUNetGenerator: Full generator architecture (encoder + decoder + output heads)
- PatchGAN70: 70x70 receptive field discriminator for adversarial training

V3 Models (Phase 16):
- MultiStainInputProjection: FiLM-conditioned input projection for variable inputs
- ModalityDropout: Training-time modality dropout for robustness
- V3Generator: Variable-input generator for IHC -> multiplex prediction

Cross-Marker Attention (Phase 12):
- CrossMarkerConfig: Configuration for cross-marker attention modules
- CrossMarkerAttention: Stage 1 bottleneck attention for inter-marker modeling
- OutputRefinementModule: Stage 2 output refinement via cross-channel attention

Cross-Stain Attention (Phase 17):
- CrossStainAttentionConfig: Configuration for cross-stain attention module
- CrossStainAttention: Self-attention for input stain fusion

Factory Functions:
- create_generator: Create a configured generator instance
- create_discriminator: Create a configured discriminator instance
- create_model_pair: Create matched generator-discriminator pair
- create_v3_generator: Create V3Generator for variable-input translation
- get_model_info: Get model parameter counts and memory estimates
"""

from multiplex.models.attention import AttentionGate
from multiplex.models.cross_marker import (
    CrossMarkerAttention,
    CrossMarkerConfig,
    OutputRefinementModule,
)
from multiplex.models.cross_stain_attention import (
    CrossStainAttention,
    CrossStainAttentionConfig,
)
from multiplex.models.decoder import DecoderBlock, MultiHeadOutput
from multiplex.models.discriminator import PatchGAN70
from multiplex.models.encoder import ConvNeXtEncoder
from multiplex.models.factory import (
    create_discriminator,
    create_generator,
    create_model_pair,
    create_v3_generator,
    create_v3_generator_with_attention,
    get_model_info,
)
from multiplex.models.generator import AttentionUNetGenerator
from multiplex.models.multi_stain_encoder import (
    ModalityDropout,
    MultiStainInputProjection,
)
from multiplex.models.v3_generator import V3Generator

__all__ = [
    # Core modules
    "ConvNeXtEncoder",
    "AttentionGate",
    "DecoderBlock",
    "MultiHeadOutput",
    # Models
    "AttentionUNetGenerator",
    "PatchGAN70",
    # V3 models (Phase 16)
    "MultiStainInputProjection",
    "ModalityDropout",
    "V3Generator",
    # Cross-marker attention (Phase 12)
    "CrossMarkerConfig",
    "CrossMarkerAttention",
    "OutputRefinementModule",
    # Cross-stain attention (Phase 17)
    "CrossStainAttention",
    "CrossStainAttentionConfig",
    # Factory functions
    "create_generator",
    "create_discriminator",
    "create_model_pair",
    "create_v3_generator",
    "create_v3_generator_with_attention",
    "get_model_info",
]
