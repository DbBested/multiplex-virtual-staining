"""Factory functions for model creation.

This module provides convenient functions to create properly configured
generator and discriminator models for the virtual staining task.

The factory functions abstract away the default configurations while
allowing customization when needed.
"""

from typing import Dict, Optional, Tuple

import torch.nn as nn

from multiplex.models.cross_marker import CrossMarkerConfig
from multiplex.models.cross_stain_attention import CrossStainAttentionConfig
from multiplex.models.discriminator import PatchGAN70
from multiplex.models.generator import AttentionUNetGenerator
from multiplex.models.v3_generator import V3Generator


def create_generator(
    in_channels: int = 1,
    num_markers: int = 5,
    encoder_name: str = "convnext_base.fb_in22k_ft_in1k",
    pretrained: bool = True,
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64),
    use_checkpoint: bool = False,
    use_attention: bool = True,
    cross_marker_config: Optional[CrossMarkerConfig] = None,
) -> AttentionUNetGenerator:
    """Create an Attention U-Net generator for BF->marker prediction.

    Factory function that creates a properly configured generator with
    sensible defaults for the virtual staining task.

    Args:
        in_channels: Input channels (1 for grayscale BF). Default 1.
        num_markers: Number of output marker channels. Default 5.
        encoder_name: timm encoder model name. Default uses ConvNeXt-Base
            with ImageNet-22k pretraining fine-tuned on ImageNet-1k.
        pretrained: Whether to load pretrained encoder weights. Default True.
        decoder_channels: Decoder channel widths at each level.
            Default (512, 256, 128, 64) produces final features with 64 channels.
        use_checkpoint: Enable gradient checkpointing for memory efficiency.
            Recommended for batch size > 2 at 512x512 resolution. Default False.
        use_attention: Enable attention gates in decoder skip connections.
            When False, creates a standard U-Net (pix2pix-style). Default True.
        cross_marker_config: Optional configuration for cross-marker attention.
            When provided, enables Stage 1 (bottleneck attention) and/or
            Stage 2 (output refinement) based on config settings. Default None
            disables cross-marker attention for backward compatibility.

    Returns:
        Configured AttentionUNetGenerator instance.

    Example:
        >>> generator = create_generator(num_markers=5, pretrained=True)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> y = generator(x)
        >>> y.shape
        torch.Size([1, 5, 512, 512])

        >>> # With cross-marker attention
        >>> from multiplex.models import CrossMarkerConfig
        >>> config = CrossMarkerConfig()
        >>> gen_cma = create_generator(pretrained=True, cross_marker_config=config)
    """
    return AttentionUNetGenerator(
        in_channels=in_channels,
        num_markers=num_markers,
        encoder_name=encoder_name,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        use_checkpoint=use_checkpoint,
        use_attention=use_attention,
        cross_marker_config=cross_marker_config,
    )


def create_discriminator(
    in_channels: int = 6,
    ndf: int = 64,
    n_layers: int = 3,
    use_spectral_norm: bool = True,
) -> PatchGAN70:
    """Create a 70x70 PatchGAN discriminator.

    Factory function that creates a properly configured discriminator
    for adversarial training in the virtual staining task.

    Args:
        in_channels: Input channels (BF + markers). Default 6 (1 + 5).
        ndf: Base number of discriminator filters. Default 64.
        n_layers: Number of conv layers. Default 3 gives 70x70 receptive field.
        use_spectral_norm: Apply spectral normalization for stability. Default True.

    Returns:
        Configured PatchGAN70 instance.

    Example:
        >>> discriminator = create_discriminator()
        >>> bf = torch.randn(1, 1, 512, 512)
        >>> markers = torch.randn(1, 5, 512, 512)
        >>> out = discriminator(bf, markers)
        >>> out.shape
        torch.Size([1, 1, 62, 62])
    """
    return PatchGAN70(
        in_channels=in_channels,
        ndf=ndf,
        n_layers=n_layers,
        use_spectral_norm=use_spectral_norm,
    )


def create_model_pair(
    num_markers: int = 5,
    pretrained: bool = True,
    use_checkpoint: bool = False,
    cross_marker_config: Optional[CrossMarkerConfig] = None,
) -> Tuple[AttentionUNetGenerator, PatchGAN70]:
    """Create matched generator-discriminator pair for training.

    Factory function that creates a compatible generator-discriminator pair
    with matching channel configurations for adversarial training.

    Args:
        num_markers: Number of marker channels. Default 5.
        pretrained: Load pretrained encoder weights. Default True.
        use_checkpoint: Enable gradient checkpointing. Default False.
        cross_marker_config: Optional configuration for cross-marker attention.
            When provided, enables cross-marker attention in the generator.
            Default None disables cross-marker attention.

    Returns:
        Tuple of (generator, discriminator) with compatible configurations.

    Example:
        >>> generator, discriminator = create_model_pair(num_markers=5)
        >>> bf = torch.randn(1, 1, 512, 512)
        >>> fake_markers = generator(bf)
        >>> d_score = discriminator(bf, fake_markers)

        >>> # With cross-marker attention
        >>> from multiplex.models import CrossMarkerConfig
        >>> config = CrossMarkerConfig()
        >>> gen, disc = create_model_pair(cross_marker_config=config)
    """
    generator = create_generator(
        num_markers=num_markers,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
        cross_marker_config=cross_marker_config,
    )
    discriminator = create_discriminator(
        in_channels=1 + num_markers,  # BF (1) + markers (num_markers)
    )
    return generator, discriminator


def get_model_info(model: nn.Module) -> Dict[str, float]:
    """Get model information: parameter counts, memory estimate.

    Utility function to inspect model size and estimate memory requirements.

    Args:
        model: PyTorch model to inspect.

    Returns:
        Dictionary with keys:
        - 'total_params': Total number of parameters
        - 'trainable_params': Number of trainable parameters
        - 'memory_mb': Rough memory estimate in MB (params + grads in float32)

    Example:
        >>> generator = create_generator()
        >>> info = get_model_info(generator)
        >>> print(f"Parameters: {info['total_params']:,}")
        Parameters: 110,234,567
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Rough memory estimate: 4 bytes per param (float32) * 2 (params + grads)
    memory_mb = total_params * 4 * 2 / 1e6
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "memory_mb": memory_mb,
    }


def create_v3_generator(
    max_input_channels: int = 4,
    num_output_targets: int = 3,
    encoder_name: str = "convnext_base.fb_in22k_ft_in1k",
    pretrained: bool = True,
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64),
    use_checkpoint: bool = False,
    use_attention: bool = True,
    use_modality_dropout: bool = True,
    p_ihc_only: float = 0.3,
    p_ihc_h: float = 0.4,
) -> V3Generator:
    """Create V3Generator for DeepLIIF variable-input translation.

    Factory function for the v3 architecture that handles variable
    input stain configurations (IHC-only or IHC+Hematoxylin).

    Args:
        max_input_channels: Max input channels (4 = IHC RGB + H).
        num_output_targets: Number of outputs (3 = DAPI, Lap2, Marker).
        encoder_name: timm encoder model name.
        pretrained: Load pretrained encoder weights.
        decoder_channels: Decoder channel widths.
        use_checkpoint: Enable gradient checkpointing.
        use_attention: Use attention gates in decoder.
        use_modality_dropout: Apply modality dropout during training.
        p_ihc_only: Probability of IHC-only config during training.
        p_ihc_h: Probability of IHC+H config during training.

    Returns:
        Configured V3Generator instance.

    Example:
        >>> generator = create_v3_generator(pretrained=True)
        >>> x = torch.randn(1, 4, 512, 512)  # IHC + Hematoxylin
        >>> y = generator(x)
        >>> y.shape
        torch.Size([1, 3, 512, 512])
    """
    return V3Generator(
        max_input_channels=max_input_channels,
        num_output_targets=num_output_targets,
        encoder_name=encoder_name,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        use_checkpoint=use_checkpoint,
        use_attention=use_attention,
        use_modality_dropout=use_modality_dropout,
        p_ihc_only=p_ihc_only,
        p_ihc_h=p_ihc_h,
    )


def create_v3_generator_with_attention(
    # Base V3Generator params
    pretrained: bool = True,
    use_checkpoint: bool = False,
    use_modality_dropout: bool = True,
    p_ihc_only: float = 0.3,
    p_ihc_h: float = 0.4,
    # Cross-stain attention params
    use_cross_stain_attention: bool = True,
    cross_stain_num_heads: int = 4,
    cross_stain_dropout: float = 0.1,
    cross_stain_gated_fusion: bool = True,
    # Cross-marker attention params
    use_cross_marker_attention: bool = True,
    cross_marker_stage1: bool = True,
    cross_marker_stage2: bool = True,
    cross_marker_num_heads: int = 8,
    cross_marker_stage2_hidden: int = 64,
) -> V3Generator:
    """Create V3Generator with full attention configuration.

    Convenience factory that creates V3Generator with:
    - CrossStainAttention for input fusion (if enabled)
    - CrossMarkerAttention for output consistency (if enabled)

    This is the recommended way to create a fully-featured V3Generator
    for training with all Phase 17 attention enhancements.

    Args:
        pretrained: Load pretrained encoder weights. Default True.
        use_checkpoint: Enable gradient checkpointing for memory efficiency.
            Recommended for batch size > 2 at 512x512 resolution. Default False.
        use_modality_dropout: Enable modality dropout during training.
            Teaches model to handle missing Hematoxylin channel. Default True.
        p_ihc_only: Probability of IHC-only config during training. Default 0.3.
        p_ihc_h: Probability of IHC+H config during training. Default 0.4.
        use_cross_stain_attention: Enable cross-stain attention for input
            fusion after projection. Default True.
        cross_stain_num_heads: Number of attention heads for cross-stain.
            embed_dim (64) must be divisible by this. Default 4.
        cross_stain_dropout: Dropout for cross-stain attention. Default 0.1.
        cross_stain_gated_fusion: Use gated fusion in cross-stain attention.
            More stable training than simple residual. Default True.
        use_cross_marker_attention: Enable cross-marker attention for output
            consistency. Controls both Stage 1 and Stage 2. Default True.
        cross_marker_stage1: Enable bottleneck attention (Stage 1).
            Only applies if use_cross_marker_attention=True. Default True.
        cross_marker_stage2: Enable output refinement (Stage 2).
            Only applies if use_cross_marker_attention=True. Default True.
        cross_marker_num_heads: Attention heads for Stage 1. Default 8.
        cross_marker_stage2_hidden: Hidden dim for Stage 2 projection. Default 64.

    Returns:
        Configured V3Generator instance with attention modules.

    Example:
        >>> model = create_v3_generator_with_attention(pretrained=True)
        >>> x = torch.randn(1, 4, 512, 512)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 3, 512, 512])

        >>> # With forward_with_features for PatchNCE
        >>> out, features = model.forward_with_features(x)
        >>> len(features)
        4

        >>> # Disable attention for ablation
        >>> model_no_attn = create_v3_generator_with_attention(
        ...     use_cross_stain_attention=False,
        ...     use_cross_marker_attention=False,
        ... )
    """
    # Build cross-stain attention config if enabled
    cross_stain_config = None
    if use_cross_stain_attention:
        cross_stain_config = CrossStainAttentionConfig(
            embed_dim=64,  # Must match MultiStainInputProjection output
            num_heads=cross_stain_num_heads,
            dropout=cross_stain_dropout,
            use_gated_fusion=cross_stain_gated_fusion,
        )

    # Build cross-marker attention config if enabled
    cross_marker_config = None
    if use_cross_marker_attention:
        cross_marker_config = CrossMarkerConfig(
            num_markers=3,  # DeepLIIF: DAPI, Lap2, Marker
            use_stage1=cross_marker_stage1,
            stage1_num_heads=cross_marker_num_heads,
            use_stage2=cross_marker_stage2,
            stage2_hidden_dim=cross_marker_stage2_hidden,
        )

    return V3Generator(
        max_input_channels=4,
        num_output_targets=3,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
        use_modality_dropout=use_modality_dropout,
        p_ihc_only=p_ihc_only,
        p_ihc_h=p_ihc_h,
        cross_stain_config=cross_stain_config,
        cross_marker_config=cross_marker_config,
    )
