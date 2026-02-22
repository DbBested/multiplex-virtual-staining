"""
PatchNCE loss for misalignment-robust training.

This module implements contrastive learning between input and output features,
enabling robust training even when BF-marker pairs have registration noise.
This is critical for the pseudo-pair training strategy.

The implementation is adapted from CUT (Park et al., ECCV 2020) for the
supervised image-to-image translation setting.

Reference:
    Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation"
    ECCV 2020. https://arxiv.org/abs/2007.15651
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PatchSampleMLP(nn.Module):
    """MLP projection head for PatchNCE patch features.

    Projects patch features to a normalized embedding space for contrastive
    learning. Uses a 2-layer MLP following SimCLR/MoCo conventions.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Dimension of output embeddings. Default 256.

    Example:
        >>> mlp = PatchSampleMLP(512, 256)
        >>> x = torch.randn(4, 512, 16, 16)  # Feature map
        >>> features, indices = mlp(x, num_patches=64)
        >>> features.shape
        torch.Size([4, 256, 64])
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, num_patches: int = 256
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample and project patch features.

        Args:
            x: Feature map of shape (B, C, H, W).
            num_patches: Number of patches to sample. If H*W <= num_patches,
                all patches are used without sampling.

        Returns:
            Tuple of:
            - Sampled and projected features of shape (B, out_channels, num_patches).
            - Patch indices of shape (num_patches,) if sampling was done, else None.
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

        patch_ids = None
        if H * W > num_patches:
            patch_ids = torch.randperm(H * W, device=x.device)[:num_patches]
            x = x[:, patch_ids, :]  # (B, num_patches, C)

        x = self.mlp(x)  # (B, num_patches, out_channels)
        x = x.permute(0, 2, 1)  # (B, out_channels, num_patches)

        return x, patch_ids


class PatchNCELoss(nn.Module):
    """Multi-layer PatchNCE loss for misalignment-robust training.

    Computes InfoNCE loss between corresponding patches of input and output
    features at multiple encoder layers. This encourages structural consistency
    even when exact pixel alignment is imperfect.

    The loss is computed as:
        L_NCE = -log(exp(q*k+ / t) / sum(exp(q*k / t)))

    where q is a query patch from the input, k+ is the corresponding positive
    patch from the output, and k are all patches (including negatives).

    Args:
        nce_layers: Encoder layer indices to use. Default [0, 2] for stages 0 and 2.
        nce_t: Temperature for InfoNCE softmax. Default 0.07.
        num_patches: Number of patches to sample per layer. Default 256.
        lambda_nce: Weight for NCE loss. Default 1.0.

    Example:
        >>> patchnce = PatchNCELoss(nce_layers=[0, 2], num_patches=64)
        >>> patchnce.init_mlp_heads((128, 256, 512, 1024))  # encoder channels
        >>> feat_q = [torch.randn(2, 128, 32, 32), torch.randn(2, 256, 16, 16),
        ...           torch.randn(2, 512, 8, 8), torch.randn(2, 1024, 4, 4)]
        >>> feat_k = [f.clone() for f in feat_q]
        >>> loss = patchnce(feat_q, feat_k)

    Reference:
        CUT (Park et al., ECCV 2020), adapted for supervised setting.
    """

    def __init__(
        self,
        nce_layers: List[int] = None,
        nce_t: float = 0.07,
        num_patches: int = 256,
        lambda_nce: float = 1.0,
    ):
        super().__init__()
        if nce_layers is None:
            nce_layers = [0, 2]
        self.nce_layers = nce_layers
        self.nce_t = nce_t
        self.num_patches = num_patches
        self.lambda_nce = lambda_nce

        self.mlp_heads = nn.ModuleDict()
        self.cross_entropy = nn.CrossEntropyLoss()
        self._initialized = False

        # Store expected encoder channels for deferred initialization
        self._channels: Optional[Tuple[int, ...]] = None

        # Register a dummy buffer to track device
        self.register_buffer("_device_tracker", torch.zeros(1))

    def init_mlp_heads(self, channels: Tuple[int, ...]) -> None:
        """Initialize MLP heads based on encoder channel configuration.

        Must be called before forward() to set up projection heads
        for each layer specified in nce_layers.

        Args:
            channels: Tuple of channel counts at each encoder stage.
                For example, (128, 256, 512, 1024) for ConvNeXt-Base.
        """
        self._channels = channels

        # Get device from the registered buffer
        device = self._device_tracker.device

        for idx in self.nce_layers:
            if idx < len(channels):
                self.mlp_heads[str(idx)] = PatchSampleMLP(channels[idx], 256).to(device)
        self._initialized = True

    def forward(
        self,
        feat_q: List[torch.Tensor],
        feat_k: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute PatchNCE loss across layers.

        Args:
            feat_q: Query features from input encoding (list of encoder stage outputs).
            feat_k: Key features from output encoding (list of encoder stage outputs).

        Returns:
            Total PatchNCE loss (scalar tensor).

        Raises:
            RuntimeError: If init_mlp_heads() was not called before forward().
        """
        if not self._initialized:
            raise RuntimeError("Call init_mlp_heads() before forward()")

        total_loss = torch.tensor(0.0, device=feat_q[0].device)
        num_layers = 0

        for idx in self.nce_layers:
            if str(idx) not in self.mlp_heads:
                continue

            mlp = self.mlp_heads[str(idx)]

            # Project and sample patches
            f_q, patch_ids = mlp(feat_q[idx], self.num_patches)
            # Use same patch locations for keys
            if patch_ids is not None:
                B, C, H, W = feat_k[idx].shape
                f_k_flat = feat_k[idx].permute(0, 2, 3, 1).reshape(B, H * W, C)
                f_k_sampled = f_k_flat[:, patch_ids, :]
                f_k_sampled = mlp.mlp(f_k_sampled).permute(0, 2, 1)
            else:
                f_k_sampled, _ = mlp(feat_k[idx], self.num_patches)

            loss = self._compute_nce(f_q, f_k_sampled)
            total_loss = total_loss + loss
            num_layers += 1

        return self.lambda_nce * total_loss / max(num_layers, 1)

    def _compute_nce(self, f_q: torch.Tensor, f_k: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss between query and key features.

        Args:
            f_q: Query features of shape (B, C, S) where S is num_patches.
            f_k: Key features of shape (B, C, S).

        Returns:
            InfoNCE loss (scalar tensor).
        """
        B, C, S = f_q.shape

        # L2 normalize features
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)

        # Positive scores: corresponding patches (diagonal)
        l_pos = (f_q * f_k).sum(dim=1, keepdim=True)  # (B, 1, S)

        # Negative scores: all other patches
        l_neg = torch.bmm(f_q.permute(0, 2, 1), f_k)  # (B, S, S)

        # Mask out positives (diagonal)
        diagonal_mask = torch.eye(S, device=f_q.device, dtype=torch.bool)
        l_neg = l_neg.masked_fill(diagonal_mask, float("-inf"))

        # Logits: [pos, neg1, neg2, ...]
        logits = torch.cat([l_pos.permute(0, 2, 1), l_neg], dim=2) / self.nce_t

        # Target: always index 0 (positive)
        targets = torch.zeros(B * S, dtype=torch.long, device=f_q.device)

        return self.cross_entropy(logits.view(B * S, -1), targets)
