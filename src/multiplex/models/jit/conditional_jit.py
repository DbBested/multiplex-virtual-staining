"""Conditional JiT model for paired image-to-image translation.

This module implements the complete ConditionalJiT model that combines:
- Source encoder (V3 ConvNeXt) for conditioning
- Patch embedding for noisy target images
- Transformer blocks with cross-attention
- Unpatchify to reconstruct output

The model is designed for paired I2I translation with flow matching:
- Input: noisy target + timestep + source image + availability config
- Output: predicted clean target (x-prediction formulation)

Reference:
    JiT: Towards Scaling Image Tokenizers (arXiv:2511.13720)
    DiT: Scalable Diffusion Models with Transformers
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from multiplex.models.jit.blocks import ConditionalJiTBlock
from multiplex.models.jit.decoder import ConvUpsampleDecoder, ProgressiveDecoder
from multiplex.models.jit.embeddings import (
    BottleneckPatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)
from multiplex.models.jit.source_encoder import SourceEncoder


class ConditionalJiT(nn.Module):
    """Conditional JiT model for paired image-to-image translation.

    A diffusion transformer adapted for paired I2I translation with:
    - Cross-attention conditioning from source encoder
    - AdaLN-Zero modulation from timestep + availability
    - Flow matching compatible (velocity or x-prediction)

    Architecture:
        1. BottleneckPatchEmbed: Patchify noisy target (B, 3, 512, 512) -> (B, 256, D)
        2. SourceEncoder: Encode source (B, 3-4, 512, 512) -> (B, 256, D)
        3. Add position embedding to target tokens
        4. Compute conditioning: timestep_embed + availability_embed
        5. Transformer blocks with cross-attention
        5.5. Optional MarkerGNN refinement (Phase 23: after final_norm, before unpatchify)
        6. Final norm + unpatchify to (B, 3, 512, 512)

    Args:
        img_size: Input/output image size. Default 512.
        patch_size: Size of each patch. Default 32.
        in_chans: Number of noisy target channels. Default 3.
        out_chans: Number of output channels (DAPI, Lap2, Marker). Default 3.
        hidden_size: Transformer dimension. Default 768 (JiT-B).
        depth: Number of transformer blocks. Default 12.
        num_heads: Number of attention heads. Default 12.
        mlp_ratio: FFN expansion ratio. Default 4.0.
        freeze_encoder: Whether to freeze source encoder. Default True.
        use_marker_gnn: Whether to enable MarkerGNN refinement (Phase 23).
            When False (default), model behavior is identical to pre-Phase 23.
        marker_gnn_node_dim: Per-node feature dimension for MarkerGNN. Default 192.
        marker_gnn_heads: Number of attention heads per GATv2 layer. Default 4.
        marker_gnn_layers: Number of GNN message-passing layers. Default 2.
        marker_gnn_dropout: Dropout rate in GNN layers. Default 0.1.
        marker_gnn_bio_prior: Whether to initialize edge weights with biological
            priors. Default True.
        use_gradient_checkpointing: Whether to enable gradient checkpointing for
            transformer blocks. Trades compute for memory. Default False.

    Attributes:
        patch_embed: Bottleneck patch embedding for noisy target.
        source_encoder: V3 ConvNeXt wrapper for source encoding.
        pos_embed: Learnable position embedding.
        t_embedder: Timestep embedding for flow matching.
        avail_embedder: Availability config embedding.
        blocks: Stack of conditional transformer blocks.
        final_norm: Final RMSNorm before output.
        unpatchify: Linear projection from tokens to pixels.

    Example:
        >>> model = ConditionalJiT(depth=12, hidden_size=768)
        >>> noisy = torch.randn(2, 3, 512, 512)
        >>> t = torch.rand(2)
        >>> source = torch.randn(2, 4, 512, 512)
        >>> avail = torch.tensor([1, 1])
        >>> output = model(noisy, t, source, avail)
        >>> output.shape
        torch.Size([2, 3, 512, 512])
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 32,
        in_chans: int = 3,
        out_chans: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        freeze_encoder: bool = True,
        # MarkerGNN (Phase 23)
        use_marker_gnn: bool = False,
        marker_gnn_node_dim: int = 192,
        marker_gnn_heads: int = 4,
        marker_gnn_layers: int = 2,
        marker_gnn_dropout: float = 0.1,
        marker_gnn_bio_prior: bool = True,
        # Gradient checkpointing (Phase 25)
        use_gradient_checkpointing: bool = False,
        nonzero_init: bool = False,
        # CNN decoder (replaces linear unpatchify)
        use_conv_upsample: bool = False,
        use_cnn_decoder: bool = False,
        # Bridge matching (source→target flow)
        use_bridge: bool = False,
        source_chans: int = 4,
    ):
        """Initialize ConditionalJiT.

        Args:
            img_size: Input/output image size.
            patch_size: Size of each patch.
            in_chans: Number of noisy target channels.
            out_chans: Number of output channels.
            hidden_size: Transformer dimension.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: FFN expansion ratio.
            freeze_encoder: Whether to freeze source encoder.
            use_marker_gnn: Whether to enable MarkerGNN refinement.
            marker_gnn_node_dim: Per-node feature dimension for MarkerGNN.
            marker_gnn_heads: Attention heads per GATv2 layer.
            marker_gnn_layers: Number of GNN message-passing layers.
            marker_gnn_dropout: Dropout in GNN layers.
            marker_gnn_bio_prior: Initialize edge weights with biological priors.
            use_gradient_checkpointing: Enable gradient checkpointing for
                transformer blocks. Trades compute for memory.
            nonzero_init: Use small random init for unpatchify instead of
                zero init. Useful for velocity prediction mode.
            use_conv_upsample: Use bilinear+conv decoder instead of linear
                unpatchify. Shares spatial info across patch boundaries.
            use_cnn_decoder: Use progressive CNN decoder (PixelShuffle) instead
                of linear unpatchify. Produces much finer spatial detail.
            use_bridge: Enable bridge matching (source→target flow).
                Adds a learned projection from source to target space.
            source_chans: Number of source channels (for bridge projection).
        """
        super().__init__()
        self._nonzero_init = nonzero_init

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing

        num_patches = (img_size // patch_size) ** 2  # 256 for 512/32

        # Patch embedding for noisy target
        self.patch_embed = BottleneckPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size,
        )

        # Source encoder (reuses V3 ConvNeXt)
        self.source_encoder = SourceEncoder(
            hidden_size=hidden_size,
            pretrained=True,
            freeze_encoder=freeze_encoder,
        )

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Timestep embedding for flow matching
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Availability config embedding (3 configs: IHC-only, IHC+H, full)
        self.avail_embedder = nn.Embedding(3, hidden_size)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            ConditionalJiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=True,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.final_norm = RMSNorm(hidden_size)

        # Output decoder: conv_upsample > cnn_decoder > linear unpatchify
        self.use_cnn_decoder = use_cnn_decoder or use_conv_upsample
        if use_conv_upsample:
            self.decoder = ConvUpsampleDecoder(
                hidden_size=hidden_size,
                out_chans=out_chans,
                img_size=img_size,
                patch_size=patch_size,
            )
            self.unpatchify = None
        elif use_cnn_decoder:
            self.decoder = ProgressiveDecoder(
                hidden_size=hidden_size,
                out_chans=out_chans,
                img_size=img_size,
                patch_size=patch_size,
            )
            self.unpatchify = None
        else:
            # Linear unpatchify (original)
            self.decoder = None
            self.unpatchify = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_chans,
            )

        # Patch boundary refinement: smooths checkerboard artifacts from
        # independent per-patch projection. Two 3x3 convs with residual
        # connection, zero-initialized so training starts as identity.
        self.output_refine = nn.Sequential(
            nn.Conv2d(out_chans, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),
        )

        # Bridge matching: learned projection from source to target space
        self.use_bridge = use_bridge
        if use_bridge:
            self.source_to_target = nn.Sequential(
                nn.Conv2d(source_chans, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, out_chans, 3, padding=1),
            )

        # Initialize weights
        self._init_weights()

        # Optional MarkerGNN (GNN-01 through GNN-04, Phase 23)
        # Conditional import: MarkerGNN is only loaded when use_marker_gnn=True.
        # MarkerGNN has its own initialization (gate=0 for identity at init).
        self.use_marker_gnn = use_marker_gnn
        if use_marker_gnn:
            from multiplex.models.marker_gnn import MarkerGNN
            self.marker_gnn = MarkerGNN(
                hidden_size=hidden_size,
                node_dim=marker_gnn_node_dim,
                n_heads=marker_gnn_heads,
                n_layers=marker_gnn_layers,
                dropout=marker_gnn_dropout,
                use_bio_prior=marker_gnn_bio_prior,
            )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Position embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Unpatchify init (only used when not using CNN decoder)
        if self.unpatchify is not None:
            if getattr(self, '_nonzero_init', False):
                nn.init.trunc_normal_(self.unpatchify.weight, std=0.02)
                nn.init.zeros_(self.unpatchify.bias)
            else:
                nn.init.zeros_(self.unpatchify.weight)
                nn.init.zeros_(self.unpatchify.bias)

        # Output refinement: zero-init last conv for identity at start
        if hasattr(self, 'output_refine'):
            nn.init.zeros_(self.output_refine[-1].weight)
            nn.init.zeros_(self.output_refine[-1].bias)

        # Bridge projection: small init so initial bridge start is a
        # crude approximation of the target
        if getattr(self, 'use_bridge', False) and hasattr(self, 'source_to_target'):
            for m in self.source_to_target.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        t: torch.Tensor,
        source: torch.Tensor,
        avail_config: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of ConditionalJiT.

        Args:
            noisy_target: Noisy target image of shape (B, 3, 512, 512).
            t: Timestep in [0, 1] of shape (B,).
            source: Source IHC image of shape (B, 3-4, 512, 512).
            avail_config: Availability config of shape (B,).
                - 0: IHC-only
                - 1: IHC+H
                - 2: Full

        Returns:
            Predicted clean target of shape (B, 3, 512, 512).
        """
        B = noisy_target.shape[0]

        # Encode source image to tokens: (B, 256, hidden_size)
        source_tokens = self.source_encoder(source, avail_config)

        # Patchify noisy target: (B, 3, 512, 512) -> (B, 256, hidden_size)
        x = self.patch_embed(noisy_target)

        # Add position embedding
        x = x + self.pos_embed

        # Compute conditioning: timestep + availability
        c = self.t_embedder(t) + self.avail_embedder(avail_config)

        # Transformer blocks with cross-attention
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch_checkpoint(
                    block, x, source_tokens, c, use_reentrant=False
                )
            else:
                x = block(x, source_tokens, c)

        # Final normalization
        x = self.final_norm(x)

        # MarkerGNN refinement (GNN-04: after transformer blocks, before output)
        if self.use_marker_gnn:
            x = self.marker_gnn(x)

        # Decode tokens to image
        if self.decoder is not None:
            # CNN decoder: (B, N, D) -> (B, out_chans, img_size, img_size)
            x = self.decoder(x)
        else:
            # Linear unpatchify + reshape (original)
            x = self.unpatchify(x)
            h = w = self.img_size // self.patch_size
            x = rearrange(
                x,
                'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=h,
                w=w,
                p1=self.patch_size,
                p2=self.patch_size,
                c=self.out_chans,
            )

        # Smooth patch boundary artifacts (zero-init residual)
        x = x + self.output_refine(x)

        return x

    def get_bridge_start(self, source: torch.Tensor) -> torch.Tensor:
        """Project source to target space for bridge matching.

        Args:
            source: Source image (B, source_chans, H, W).

        Returns:
            Projected source in target space (B, out_chans, H, W).
        """
        return self.source_to_target(source)
