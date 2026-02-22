"""V3Generator for variable-input multi-stain to multiplex prediction.

This module implements the V3 generator architecture that combines:
- MultiStainInputProjection with FiLM conditioning for variable input handling
- ModalityDropout for training-time input masking
- ConvNeXt-Base pretrained encoder for multi-scale feature extraction
- Attention-gated decoder with skip connections
- DeepLIIFMultiHeadOutput for DAPI, Lap2, Marker predictions

The V3Generator accepts variable input configurations (IHC-only or IHC+Hematoxylin)
and predicts 3 fluorescence marker channels simultaneously.

Architecture:
    Input: (B, 3, H, W) IHC-only or (B, 4, H, W) IHC+Hematoxylin
    Input Projection: FiLM-conditioned projection to 64 channels
    Encoder: ConvNeXt-Base [128, 256, 512, 1024] channels at 1/4, 1/8, 1/16, 1/32 resolution
    Decoder: [512, 256, 128, 64] with attention gates at each skip
    Output: (B, 3, H, W) DAPI, Lap2, Marker predictions

Reference:
    Architecture inspired by Attention U-Net (Oktay et al., 2018) with
    modern ConvNeXt encoder backbone (Liu et al., 2022) and FiLM conditioning
    for variable input handling (Perez et al., 2018).
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from multiplex.models.decoder import DecoderBlock
from multiplex.models.encoder import ConvNeXtEncoder
from multiplex.models.multi_stain_encoder import (
    ModalityDropout,
    MultiStainInputProjection,
)

if TYPE_CHECKING:
    from multiplex.models.cross_stain_attention import CrossStainAttentionConfig
    from multiplex.models.cross_marker import CrossMarkerConfig


class DeepLIIFMultiHeadOutput(nn.Module):
    """Multi-head output for DeepLIIF-style DAPI, Lap2, Marker predictions.

    Similar to MultiHeadOutput but with named targets for DeepLIIF outputs.
    Each head produces a single-channel prediction, concatenated in order:
    DAPI (channel 0), Lap2 (channel 1), Marker (channel 2).

    Attributes:
        heads: ModuleDict mapping target name to Conv2d head.
        TARGET_NAMES: Class attribute listing target names in output order.

    Example:
        >>> output = DeepLIIFMultiHeadOutput(in_channels=64)
        >>> x = torch.randn(1, 64, 512, 512)
        >>> out = output(x)
        >>> out.shape
        torch.Size([1, 3, 512, 512])
    """

    TARGET_NAMES = ["DAPI", "Lap2", "Marker"]

    def __init__(self, in_channels: int, num_targets: int = 3):
        """Initialize DeepLIIFMultiHeadOutput.

        Args:
            in_channels: Number of input channels from final decoder layer.
            num_targets: Number of output targets. Default 3 for DeepLIIF.
        """
        super().__init__()

        self.num_targets = num_targets

        # Create named heads in ModuleDict for interpretability
        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(in_channels, 1, kernel_size=1)
                for name in self.TARGET_NAMES[:num_targets]
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all output heads and concatenate in target order.

        Args:
            x: Decoder features, shape (B, in_channels, H, W).

        Returns:
            Multi-target prediction, shape (B, num_targets, H, W).
            Channels correspond to DAPI, Lap2, Marker in that order.
        """
        outputs = [self.heads[name](x) for name in self.TARGET_NAMES[: self.num_targets]]
        return torch.cat(outputs, dim=1)

    def get_target_names(self) -> List[str]:
        """Get target names in output channel order.

        Returns:
            List of target names: ['DAPI', 'Lap2', 'Marker'].
        """
        return self.TARGET_NAMES[: self.num_targets]


class V3Generator(nn.Module):
    """V3 Generator for variable-input multi-stain to multiplex prediction.

    Full generator architecture combining MultiStainInputProjection with
    FiLM conditioning, ConvNeXt encoder, Attention U-Net decoder, and
    DeepLIIF multi-head output.

    This architecture accepts variable input configurations:
    - IHC-only (3 channels): Just the IHC RGB image
    - IHC+Hematoxylin (4 channels): IHC RGB + Hematoxylin grayscale

    The model learns to predict DAPI, Lap2, and Marker channels from either
    input configuration through modality dropout during training.

    Attributes:
        modality_dropout: Optional ModalityDropout for training-time masking.
        input_proj: MultiStainInputProjection with FiLM conditioning.
        encoder: ConvNeXt-Base encoder with multi-scale feature extraction.
        bottleneck: Convolutional block at the encoder output.
        decoders: ModuleList of DecoderBlock modules.
        final_upsample: Upsampling to full input resolution.
        output_heads: DeepLIIFMultiHeadOutput for 3-target prediction.
        use_checkpoint: Whether gradient checkpointing is enabled.
        use_attention: Whether attention gates are enabled in decoder.

    Example:
        >>> model = V3Generator(pretrained=True)
        >>> x_ihc = torch.randn(1, 3, 512, 512)  # IHC-only
        >>> y = model(x_ihc, avail_config=torch.tensor([0]))
        >>> y.shape
        torch.Size([1, 3, 512, 512])

        >>> x_full = torch.randn(1, 4, 512, 512)  # IHC + Hematoxylin
        >>> y = model(x_full, avail_config=torch.tensor([1]))
        >>> y.shape
        torch.Size([1, 3, 512, 512])

        >>> # Auto-infer config from input channels
        >>> y = model(x_full)
        >>> y.shape
        torch.Size([1, 3, 512, 512])
    """

    def __init__(
        self,
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
        cross_stain_config: Optional["CrossStainAttentionConfig"] = None,
        cross_marker_config: Optional["CrossMarkerConfig"] = None,
    ):
        """Initialize V3Generator.

        Args:
            max_input_channels: Maximum input channels. Default 4 for
                IHC RGB (3) + Hematoxylin grayscale (1).
            num_output_targets: Number of output targets. Default 3 for
                DAPI, Lap2, Marker.
            encoder_name: timm model name for encoder. Default uses ConvNeXt-Base
                with ImageNet-22k pretraining fine-tuned on ImageNet-1k.
            pretrained: Whether to load pretrained ImageNet weights.
            decoder_channels: Tuple of output channels for each decoder block.
                Default (512, 256, 128, 64) produces final features with 64 channels.
            use_checkpoint: Enable gradient checkpointing to reduce memory usage.
                Recommended for batch size > 2 at 512x512 resolution.
            use_attention: Whether to use attention gates in decoder blocks.
                Default True. Set False for standard U-Net skip concatenation.
            use_modality_dropout: Whether to apply modality dropout during training.
                Default True. Teaches model to handle missing Hematoxylin channel.
            p_ihc_only: Probability of IHC-only config during training dropout.
                Default 0.3 (30% of samples see IHC-only during training).
            p_ihc_h: Probability of IHC+H config during training dropout.
                Default 0.4 (40% of samples see IHC+H during training).
            cross_stain_config: Optional CrossStainAttentionConfig for input fusion.
                When provided, applies self-attention after input projection to
                fuse information across spatial locations. Default None (disabled).
            cross_marker_config: Optional CrossMarkerConfig for output consistency.
                When provided, enables Stage 1 (bottleneck attention) and/or
                Stage 2 (output refinement) for inter-marker relationship modeling.
                num_markers in config must match num_output_targets. Default None.
        """
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_output_targets = num_output_targets
        self.use_attention = use_attention
        self.use_modality_dropout = use_modality_dropout
        self.max_input_channels = max_input_channels

        # Input projection embedding dimension
        embed_dim = 64

        # 1. Modality dropout (optional, for training)
        if use_modality_dropout:
            self.modality_dropout = ModalityDropout(
                p_ihc_only=p_ihc_only, p_ihc_h=p_ihc_h
            )
        else:
            self.modality_dropout = None

        # 2. Input projection with FiLM conditioning
        self.input_proj = MultiStainInputProjection(
            max_input_channels=max_input_channels, embed_dim=embed_dim
        )

        # 2.5. Cross-stain attention (optional, for input fusion)
        self.cross_stain_attention = None
        if cross_stain_config is not None:
            from multiplex.models.cross_stain_attention import CrossStainAttention

            self.cross_stain_attention = CrossStainAttention(cross_stain_config)

        # 3. Encoder: ConvNeXt-Base with pretrained weights
        # Takes projected 64-channel input
        self.encoder = ConvNeXtEncoder(
            in_channels=embed_dim,
            pretrained=pretrained,
            model_name=encoder_name,
        )

        # Get encoder channel configuration: [128, 256, 512, 1024]
        encoder_channels = self.encoder.get_channels()

        # 4. Bottleneck: process deepest encoder features
        # Input: 1024 channels at 1/32 resolution
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                encoder_channels[-1],
                encoder_channels[-1],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # 5. Decoder blocks with attention-gated skip connections
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
                    use_attention=use_attention,
                )
            )
            in_ch = out_ch

        # 6. Final upsampling: decoder output is at 1/4 resolution, need full res
        # After 4 decoder blocks: 1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2
        # But we start at 1/32 and skip to 1/16, 1/8, 1/4, so output is at 1/4
        # Need 4x upsample to get to full resolution
        self.final_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # 7. Output heads: DeepLIIF multi-head for DAPI, Lap2, Marker
        self.output_heads = DeepLIIFMultiHeadOutput(
            in_channels=decoder_channels[-1], num_targets=num_output_targets
        )

        # 8. Cross-marker attention (optional, for output consistency)
        self.cross_marker_attention = None
        self.output_refinement = None
        if cross_marker_config is not None:
            from multiplex.models.cross_marker import (
                CrossMarkerAttention,
                OutputRefinementModule,
            )

            assert cross_marker_config.num_markers == num_output_targets, (
                f"CrossMarkerConfig.num_markers ({cross_marker_config.num_markers}) "
                f"must match num_output_targets ({num_output_targets})"
            )
            if cross_marker_config.use_stage1:
                self.cross_marker_attention = CrossMarkerAttention(cross_marker_config)
            if cross_marker_config.use_stage2:
                self.output_refinement = OutputRefinementModule(cross_marker_config)

    def forward(
        self,
        x: torch.Tensor,
        avail_config: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: variable-input to multi-target prediction.

        Args:
            x: Input tensor, shape (B, C, H, W) where C is 3 or 4.
                H, W should be divisible by 32 for proper encoder operation.
            avail_config: Optional configuration indices, shape (B,).
                Values: 0=IHC-only, 1=IHC+H, 2=full.
                If None, automatically inferred from input channels.

        Returns:
            Multi-target prediction, shape (B, num_output_targets, H, W).
            Same spatial resolution as input. Channels are DAPI, Lap2, Marker.
        """
        B, C, H, W = x.shape

        # Handle modality dropout during training
        if self.modality_dropout is not None and self.training:
            # Pad to 4 channels if needed for dropout
            if C < self.max_input_channels:
                padding = torch.zeros(
                    B,
                    self.max_input_channels - C,
                    H,
                    W,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, padding], dim=1)
            x, avail_config = self.modality_dropout(x)

        # Infer config from input channels if not provided
        if avail_config is None:
            # 3 channels -> IHC-only (config=0)
            # 4 channels -> IHC+H (config=1)
            if C == 3:
                avail_config = torch.zeros(B, device=x.device, dtype=torch.long)
            else:
                avail_config = torch.ones(B, device=x.device, dtype=torch.long)

        # 1. Input projection with FiLM conditioning
        h = self.input_proj(x, avail_config)

        # 1.5. Cross-stain attention (if enabled)
        if self.cross_stain_attention is not None:
            h = self.cross_stain_attention(h, avail_config)

        # 2. Encode: extract multi-scale features
        if self.use_checkpoint and self.training:
            features = checkpoint(self._encode, h, use_reentrant=False)
        else:
            features = self._encode(h)

        # features is list of 4 tensors:
        # [0]: (B, 128, H/4, W/4)   - stage 0
        # [1]: (B, 256, H/8, W/8)   - stage 1
        # [2]: (B, 512, H/16, W/16) - stage 2
        # [3]: (B, 1024, H/32, W/32) - stage 3

        # 3. Bottleneck: process deepest features
        x = self.bottleneck(features[-1])

        # 3.5. Cross-marker attention Stage 1 (if enabled)
        if self.cross_marker_attention is not None:
            x = self.cross_marker_attention(x)

        # 4. Decode with attention-gated skip connections
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

        # 5. Final upsample to full resolution
        x = self.final_upsample(x)

        # 6. Multi-head output
        output = self.output_heads(x)

        # 6.5. Cross-marker attention Stage 2: output refinement (if enabled)
        if self.output_refinement is not None:
            output = self.output_refinement(output)

        return output

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Wrapper for encoder forward to support checkpointing."""
        return self.encoder(x)

    def forward_with_features(
        self,
        x: torch.Tensor,
        avail_config: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning output AND encoder features for PatchNCE.

        This duplicates the forward logic but also returns the encoder features,
        which are needed for PatchNCE contrastive loss computation.

        Args:
            x: Input tensor, shape (B, C, H, W) where C is 3 or 4.
                H, W should be divisible by 32 for proper encoder operation.
            avail_config: Optional configuration indices, shape (B,).
                Values: 0=IHC-only, 1=IHC+H, 2=full.
                If None, automatically inferred from input channels.

        Returns:
            Tuple of:
            - output: Multi-target prediction, shape (B, num_output_targets, H, W).
            - encoder_features: List of 4 tensors from each encoder stage:
                [0]: (B, 128, H/4, W/4)
                [1]: (B, 256, H/8, W/8)
                [2]: (B, 512, H/16, W/16)
                [3]: (B, 1024, H/32, W/32)

        Example:
            >>> model = V3Generator(pretrained=False)
            >>> x = torch.randn(1, 4, 128, 128)
            >>> output, features = model.forward_with_features(x)
            >>> len(features)
            4
            >>> features[0].shape
            torch.Size([1, 128, 32, 32])
        """
        B, C, H, W = x.shape

        # Handle modality dropout during training
        if self.modality_dropout is not None and self.training:
            # Pad to 4 channels if needed for dropout
            if C < self.max_input_channels:
                padding = torch.zeros(
                    B,
                    self.max_input_channels - C,
                    H,
                    W,
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, padding], dim=1)
            x, avail_config = self.modality_dropout(x)

        # Infer config from input channels if not provided
        if avail_config is None:
            # 3 channels -> IHC-only (config=0)
            # 4 channels -> IHC+H (config=1)
            if C == 3:
                avail_config = torch.zeros(B, device=x.device, dtype=torch.long)
            else:
                avail_config = torch.ones(B, device=x.device, dtype=torch.long)

        # 1. Input projection with FiLM conditioning
        h = self.input_proj(x, avail_config)

        # 1.5. Cross-stain attention (if enabled)
        if self.cross_stain_attention is not None:
            h = self.cross_stain_attention(h, avail_config)

        # 2. Encode: extract multi-scale features
        if self.use_checkpoint and self.training:
            features = checkpoint(self._encode, h, use_reentrant=False)
        else:
            features = self._encode(h)

        # 3. Bottleneck: process deepest features
        z = self.bottleneck(features[-1])

        # 3.5. Cross-marker attention Stage 1 (if enabled)
        if self.cross_marker_attention is not None:
            z = self.cross_marker_attention(z)

        # 4. Decode with attention-gated skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(features) - 2 - i
            skip_idx = max(skip_idx, 0)
            skip = features[skip_idx]

            if self.use_checkpoint and self.training:
                z = checkpoint(decoder, z, skip, use_reentrant=False)
            else:
                z = decoder(z, skip)

        # 5. Final upsample to full resolution
        z = self.final_upsample(z)

        # 6. Multi-head output
        output = self.output_heads(z)

        # 6.5. Cross-marker attention Stage 2: output refinement (if enabled)
        if self.output_refinement is not None:
            output = self.output_refinement(output)

        return output, features

    def get_target_names(self) -> List[str]:
        """Get target names in output channel order.

        Returns:
            List of target names: ['DAPI', 'Lap2', 'Marker'].
        """
        return self.output_heads.get_target_names()

    def get_attention_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get attention maps from all decoder blocks.

        Returns:
            Dictionary mapping decoder block names to attention maps.
            Each map has shape (B, 1, H, W) with values in [0, 1].
            Returns None for blocks where forward hasn't been called
            or if attention is disabled.

        Example:
            >>> model = V3Generator(pretrained=False)
            >>> _ = model(torch.randn(1, 4, 512, 512))
            >>> attn_maps = model.get_attention_maps()
            >>> list(attn_maps.keys())
            ['decoder_0', 'decoder_1', 'decoder_2', 'decoder_3']
        """
        return {
            f"decoder_{i}": decoder.get_attention_map()
            for i, decoder in enumerate(self.decoders)
        }

    def get_all_attention_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get attention maps from all attention modules.

        Returns a comprehensive dictionary including:
        - Decoder attention gates (decoder_0, decoder_1, etc.)
        - Cross-stain attention (cross_stain) if enabled
        - Cross-marker Stage 1 (cross_marker_stage1) if enabled
        - Cross-marker Stage 2 (cross_marker_stage2) if enabled

        Returns:
            Dictionary mapping attention module names to attention maps.
            Returns None for modules that are disabled or haven't run forward.

        Example:
            >>> from multiplex.models import CrossStainAttentionConfig, CrossMarkerConfig
            >>> model = V3Generator(
            ...     pretrained=False,
            ...     cross_stain_config=CrossStainAttentionConfig(),
            ...     cross_marker_config=CrossMarkerConfig(num_markers=3),
            ... )
            >>> _ = model(torch.randn(1, 4, 64, 64))
            >>> attn_maps = model.get_all_attention_maps()
            >>> 'cross_stain' in attn_maps
            True
        """
        maps = self.get_attention_maps()

        # Add cross-stain attention if enabled
        if self.cross_stain_attention is not None:
            maps["cross_stain"] = self.cross_stain_attention.get_attention_map()

        # Add cross-marker Stage 1 attention if enabled
        if self.cross_marker_attention is not None:
            maps["cross_marker_stage1"] = self.cross_marker_attention.get_attention_map()

        # Add cross-marker Stage 2 attention if enabled
        if self.output_refinement is not None:
            maps["cross_marker_stage2"] = self.output_refinement.get_attention_map()

        return maps

    def get_encoder_channels(self) -> Tuple[int, ...]:
        """Get encoder channel configuration.

        Returns:
            Tuple of channel counts: (128, 256, 512, 1024) for ConvNeXt-Base.
        """
        return self.encoder.get_channels()

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for model components.

        Returns:
            Dictionary with parameter counts for each component and total.
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoders.parameters())
        bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters())
        output_params = sum(p.numel() for p in self.output_heads.parameters())
        input_proj_params = sum(p.numel() for p in self.input_proj.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "input_proj": input_proj_params,
            "encoder": encoder_params,
            "bottleneck": bottleneck_params,
            "decoder": decoder_params,
            "output_heads": output_params,
            "total": total_params,
        }
