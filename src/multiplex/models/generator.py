"""Attention U-Net Generator for brightfield to multi-marker prediction.

This module implements the full generator architecture that combines:
- ConvNeXt-Base pretrained encoder for multi-scale feature extraction
- Attention-gated decoder with skip connections
- Multi-head output for 5 independent marker predictions

The generator takes grayscale brightfield input and predicts 5 fluorescence
marker channels simultaneously.

Architecture:
    Input: (B, 1, H, W) grayscale brightfield
    Encoder: ConvNeXt-Base [128, 256, 512, 1024] channels at 1/4, 1/8, 1/16, 1/32 resolution
    Decoder: [512, 256, 128, 64] with attention gates at each skip
    Output: (B, 5, H, W) 5-marker predictions

Reference:
    Architecture inspired by Attention U-Net (Oktay et al., 2018) with
    modern ConvNeXt encoder backbone (Liu et al., 2022).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from multiplex.models.attention import AttentionGate
from multiplex.models.cross_marker import (
    CrossMarkerAttention,
    CrossMarkerConfig,
    OutputRefinementModule,
)
from multiplex.models.decoder import DecoderBlock, MultiHeadOutput
from multiplex.models.encoder import ConvNeXtEncoder


class AttentionUNetGenerator(nn.Module):
    """Attention U-Net with ConvNeXt-Base encoder.

    Full generator architecture for brightfield-to-fluorescence virtual staining.
    Uses pretrained ImageNet weights on the encoder for transfer learning,
    with optional attention-gated skip connections to the decoder.

    When use_attention=False, the architecture becomes a standard U-Net with
    direct skip connections (no attention gating).

    Attributes:
        encoder: ConvNeXt-Base encoder with multi-scale feature extraction.
        bottleneck: Convolutional block at the encoder output.
        decoders: ModuleList of DecoderBlock modules.
        final_upsample: Upsampling to full input resolution.
        output_heads: MultiHeadOutput for 5-marker prediction.
        use_checkpoint: Whether gradient checkpointing is enabled.
        dropout_p: Dropout probability for decoder blocks (for MC Dropout).
        use_attention: Whether attention gates are enabled in decoder.
        cross_marker_config: Optional configuration for cross-marker attention.
        cross_marker_attention: Optional Stage 1 cross-marker attention module.
        output_refinement: Optional Stage 2 output refinement module.

    Example:
        >>> model = AttentionUNetGenerator(in_channels=1, num_markers=5, pretrained=True)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> y = model(x)
        >>> y.shape
        torch.Size([1, 5, 512, 512])

        >>> # Without attention gates (standard U-Net)
        >>> model_no_attn = AttentionUNetGenerator(pretrained=False, use_attention=False)

        >>> # With cross-marker attention (Phase 12)
        >>> from multiplex.models.cross_marker import CrossMarkerConfig
        >>> config = CrossMarkerConfig()
        >>> model_cma = AttentionUNetGenerator(pretrained=True, cross_marker_config=config)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_markers: int = 5,
        encoder_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        decoder_channels: Tuple[int, ...] = (512, 256, 128, 64),
        use_checkpoint: bool = False,
        dropout_p: float = 0.0,
        use_attention: bool = True,
        cross_marker_config: Optional[CrossMarkerConfig] = None,
    ):
        """Initialize AttentionUNetGenerator.

        Args:
            in_channels: Number of input channels. Default 1 for grayscale BF.
            num_markers: Number of output marker channels. Default 5.
            encoder_name: timm model name for encoder. Default uses ConvNeXt-Base
                with ImageNet-22k pretraining fine-tuned on ImageNet-1k.
            pretrained: Whether to load pretrained ImageNet weights.
            decoder_channels: Tuple of output channels for each decoder block.
                Default (512, 256, 128, 64) produces final features with 64 channels.
            use_checkpoint: Enable gradient checkpointing to reduce memory usage.
                Recommended for batch size > 2 at 512x512 resolution.
            dropout_p: Dropout probability for decoder blocks. Used for
                MC Dropout uncertainty estimation. Default 0.0 (no dropout).
                Recommended 0.2-0.3 for uncertainty estimation.
            use_attention: Whether to use attention gates in decoder blocks.
                Default True. Set False for standard U-Net skip concatenation
                (used in ablation studies to measure attention contribution).
            cross_marker_config: Optional configuration for cross-marker attention.
                When provided, enables Stage 1 (bottleneck attention) and/or
                Stage 2 (output refinement) based on config settings. Default None
                disables cross-marker attention for backward compatibility.
        """
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_markers = num_markers
        self.dropout_p = dropout_p
        self.use_attention = use_attention

        # Encoder: ConvNeXt-Base with pretrained weights
        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            pretrained=pretrained,
            model_name=encoder_name,
        )

        # Get encoder channel configuration: [128, 256, 512, 1024]
        encoder_channels = self.encoder.get_channels()

        # Bottleneck: process deepest encoder features
        # Input: 1024 channels at 1/32 resolution
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks with attention-gated skip connections
        # Build from deepest to shallowest
        self.decoders = nn.ModuleList()
        in_ch = encoder_channels[-1]  # Start with 1024

        for i, out_ch in enumerate(decoder_channels):
            # Skip connection from corresponding encoder level
            # i=0: skip from encoder stage 2 (512 channels)
            # i=1: skip from encoder stage 1 (256 channels)
            # i=2: skip from encoder stage 0 (128 channels)
            # i=3: skip from encoder stage 0 (128 channels) - reuse shallowest
            skip_idx = len(encoder_channels) - 2 - i
            skip_idx = max(skip_idx, 0)  # Clamp to valid index
            skip_ch = encoder_channels[skip_idx]

            self.decoders.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    dropout_p=dropout_p,
                    use_attention=use_attention,
                )
            )
            in_ch = out_ch

        # Final upsampling: decoder output is at 1/4 resolution, need to get to full
        # After 4 decoder blocks: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2
        # But we start at 1/32 and skip to 1/16, 1/8, 1/4, so output is at 1/4
        # Need 4x upsample to get to full resolution
        self.final_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # Multi-head output: 5 separate 1x1 convs for each marker
        self.output_heads = MultiHeadOutput(
            in_channels=decoder_channels[-1],
            num_markers=num_markers,
        )

        # Cross-marker attention (Phase 12)
        # Enables inter-marker relationship modeling via attention
        self.cross_marker_config = cross_marker_config
        self.cross_marker_attention: Optional[CrossMarkerAttention] = None
        self.output_refinement: Optional[OutputRefinementModule] = None

        if cross_marker_config is not None:
            if cross_marker_config.use_stage1:
                self.cross_marker_attention = CrossMarkerAttention(cross_marker_config)
            if cross_marker_config.use_stage2:
                self.output_refinement = OutputRefinementModule(cross_marker_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: brightfield to multi-marker prediction.

        Args:
            x: Brightfield input, shape (B, 1, H, W).
                H, W should be divisible by 32 for proper encoder operation.

        Returns:
            Multi-marker prediction, shape (B, num_markers, H, W).
            Same spatial resolution as input.
        """
        # 1. Encode: extract multi-scale features
        if self.use_checkpoint and self.training:
            features = checkpoint(self._encode, x, use_reentrant=False)
        else:
            features = self._encode(x)

        # features is list of 4 tensors:
        # [0]: (B, 128, H/4, W/4)   - stage 0
        # [1]: (B, 256, H/8, W/8)   - stage 1
        # [2]: (B, 512, H/16, W/16) - stage 2
        # [3]: (B, 1024, H/32, W/32) - stage 3

        # 2. Bottleneck: process deepest features
        x = self.bottleneck(features[-1])

        # 2.5 Cross-marker attention Stage 1 (bottleneck)
        # Applies self-attention so each position attends to all others
        if self.cross_marker_attention is not None:
            x = self.cross_marker_attention(x)

        # 3. Decode with attention-gated skip connections
        for i, decoder in enumerate(self.decoders):
            # Get skip connection from encoder
            # i=0: use features[2] (512 ch, 1/16)
            # i=1: use features[1] (256 ch, 1/8)
            # i=2: use features[0] (128 ch, 1/4)
            # i=3: use features[0] (128 ch, 1/4) - reuse shallowest
            skip_idx = len(features) - 2 - i
            skip_idx = max(skip_idx, 0)
            skip = features[skip_idx]

            if self.use_checkpoint and self.training:
                x = checkpoint(decoder, x, skip, use_reentrant=False)
            else:
                x = decoder(x, skip)

        # 4. Final upsample to full resolution
        x = self.final_upsample(x)

        # 5. Multi-head output
        output = self.output_heads(x)

        # 5.5 Cross-marker attention Stage 2 (output refinement)
        # Each marker prediction attends to all other markers
        if self.output_refinement is not None:
            output = self.output_refinement(output)

        return output

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Wrapper for encoder forward to support checkpointing."""
        return self.encoder(x)

    def get_attention_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get attention maps from all decoder blocks.

        Returns:
            Dictionary mapping decoder block names to attention maps.
            Each map has shape (B, 1, H, W) with values in [0, 1].
            Returns None for blocks where forward hasn't been called.

        Example:
            >>> model = AttentionUNetGenerator()
            >>> _ = model(torch.randn(1, 1, 512, 512))
            >>> attn_maps = model.get_attention_maps()
            >>> list(attn_maps.keys())
            ['decoder_0', 'decoder_1', 'decoder_2', 'decoder_3']
        """
        return {
            f"decoder_{i}": decoder.get_attention_map()
            for i, decoder in enumerate(self.decoders)
        }

    def get_cross_marker_attention_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get cross-marker attention maps from Stage 1 and Stage 2.

        Returns:
            Dictionary with keys 'stage1' and 'stage2', each mapping to
            attention weights tensor or None if that stage is disabled.

            - stage1: (B, num_heads, H*W, H*W) bottleneck attention
            - stage2: (B, num_heads, H*W, H*W) output refinement attention

            Returns None for a stage if it's disabled or forward hasn't been called.

        Example:
            >>> from multiplex.models.cross_marker import CrossMarkerConfig
            >>> config = CrossMarkerConfig()
            >>> model = AttentionUNetGenerator(cross_marker_config=config)
            >>> _ = model(torch.randn(1, 1, 512, 512))
            >>> maps = model.get_cross_marker_attention_maps()
            >>> maps['stage1'].shape
            torch.Size([1, 8, 256, 256])
        """
        stage1_map = None
        stage2_map = None

        if self.cross_marker_attention is not None:
            stage1_map = self.cross_marker_attention.get_attention_map()
        if self.output_refinement is not None:
            stage2_map = self.output_refinement.get_attention_map()

        return {"stage1": stage1_map, "stage2": stage2_map}

    def get_encoder_channels(self) -> Tuple[int, ...]:
        """Get encoder channel configuration.

        Returns:
            Tuple of channel counts: (128, 256, 512, 1024) for ConvNeXt-Base.
        """
        return self.encoder.get_channels()

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for model components.

        Returns:
            Dictionary with parameter counts for encoder, decoder, and total.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoders.parameters())
        bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters())
        output_params = sum(p.numel() for p in self.output_heads.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "encoder": encoder_params,
            "bottleneck": bottleneck_params,
            "decoder": decoder_params,
            "output_heads": output_params,
            "total": total_params,
        }
