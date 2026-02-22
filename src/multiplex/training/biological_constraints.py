"""
Differentiable biological constraint losses for virtual staining training.

This module provides three constraint losses that encode known organelle spatial
relationships into the training process:

1. ExclusionLoss: Penalizes cytoplasmic markers (TOMM20, SEC61B, ACTB) appearing
   inside the nuclear region. Uses distance-weighted penalty for soft gradients.

2. ContainmentLoss: Penalizes FBL (nucleolar marker) appearing outside the
   nuclear boundary. Uses dilated boundary with distance-weighted penalty.

3. ColocalizationLoss: Encourages TOMM20-SEC61B spatial correlation via
   differentiable Manders coefficients (M1 and M2).

All losses use soft sigmoid thresholding for differentiability and kornia's
GPU-native distance transforms for efficient gradient computation.

Example:
    >>> from multiplex.training.biological_constraints import (
    ...     ExclusionLoss, ContainmentLoss, ColocalizationLoss
    ... )
    >>> import torch
    >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
    >>> excl = ExclusionLoss()
    >>> loss, loss_dict = excl(pred)
    >>> loss.backward()  # Gradients flow
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from kornia.contrib import distance_transform
    from kornia import morphology as morph
    KORNIA_AVAILABLE = True
except ImportError:
    distance_transform = None
    morph = None
    KORNIA_AVAILABLE = False


# Marker channel indices
LMNB1_IDX = 0  # Nuclear envelope
FBL_IDX = 1    # Nucleoli
TOMM20_IDX = 2  # Mitochondria
SEC61B_IDX = 3  # ER
ACTB_IDX = 4   # Cytoskeleton

CYTOPLASMIC_MARKERS = [TOMM20_IDX, SEC61B_IDX, ACTB_IDX]


def soft_threshold(
    x: torch.Tensor,
    threshold: float,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Differentiable soft thresholding via sigmoid.

    Approximates: mask = (x > threshold).float()
    But maintains gradient flow for training.

    Formula: sigma(k * (x - t)) where k is temperature

    Args:
        x: Input tensor (any shape).
        threshold: Threshold value.
        temperature: Sharpness of transition (higher = more binary-like).
                    Default 10.0, typical range 5-50.

    Returns:
        Soft mask in [0, 1] with same shape as input.

    Example:
        >>> x = torch.tensor([0.3, 0.5, 0.7])
        >>> mask = soft_threshold(x, threshold=0.5, temperature=10.0)
        >>> # mask is approximately [0.12, 0.5, 0.88]
    """
    return torch.sigmoid(temperature * (x - threshold))


def compute_otsu_threshold(x: torch.Tensor) -> float:
    """Compute Otsu threshold (non-differentiable, use with detached tensor).

    This function should be called with .detach() on the input,
    and the threshold used as a fixed parameter (not in gradient path).

    Args:
        x: Input tensor (will be flattened).

    Returns:
        Optimal threshold value from Otsu's method.

    Note:
        Returns 0.5 if Otsu computation fails (uniform histogram).
    """
    try:
        from skimage.filters import threshold_otsu
    except ImportError:
        # Fallback to simple percentile if skimage not available
        return float(x.detach().cpu().numpy().flatten().mean())

    # Convert to numpy for Otsu computation
    x_np = x.detach().cpu().numpy().flatten()

    # Handle edge cases
    if x_np.max() == x_np.min():
        return 0.5

    try:
        thresh = threshold_otsu(x_np)
    except ValueError:
        # Otsu fails if histogram is uniform
        thresh = 0.5

    return float(thresh)


def create_nuclear_mask(
    lmnb1: torch.Tensor,
    use_otsu: bool = True,
    fixed_threshold: float = 0.5,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Create soft nuclear mask from LMNB1 channel.

    Uses Otsu threshold computed from detached tensor (not in gradient path),
    then applies soft sigmoid for differentiable mask creation.

    Args:
        lmnb1: LMNB1 channel of shape (B, H, W).
        use_otsu: If True, compute per-batch Otsu threshold. If False, use fixed.
        fixed_threshold: Threshold value when use_otsu=False.
        temperature: Sigmoid temperature for soft thresholding.

    Returns:
        Soft nuclear mask of shape (B, H, W) in [0, 1].

    Example:
        >>> lmnb1 = torch.rand(4, 256, 256)
        >>> mask = create_nuclear_mask(lmnb1, use_otsu=True)
        >>> # mask is soft binary mask with gradient flow
    """
    if use_otsu:
        # Compute threshold per batch sample and average
        thresholds = []
        for b in range(lmnb1.shape[0]):
            thresh = compute_otsu_threshold(lmnb1[b])
            thresholds.append(thresh)
        threshold = sum(thresholds) / len(thresholds)
    else:
        threshold = fixed_threshold

    # Apply soft thresholding for gradient flow
    return soft_threshold(lmnb1, threshold, temperature)


@dataclass
class ExclusionConfig:
    """Configuration for ExclusionLoss.

    Attributes:
        temperature: Sigmoid temperature for soft thresholding. Default 10.0.
        margin_pixels: Distance over which penalty ramps from boundary. Default 10.
        use_otsu: Use adaptive Otsu threshold per batch. Default True.
        fixed_threshold: Nuclear threshold when use_otsu=False. Default 0.5.
        use_gt_mask: Use GT LMNB1 for mask if target provided. Default True.
    """
    temperature: float = 10.0
    margin_pixels: int = 10
    use_otsu: bool = True
    fixed_threshold: float = 0.5
    use_gt_mask: bool = True


class ExclusionLoss(nn.Module):
    """Differentiable exclusion loss for cytoplasmic markers.

    Penalizes TOMM20, SEC61B, and ACTB signal appearing inside the nuclear
    region defined by LMNB1. Uses distance-weighted penalty so violations
    deeper inside the nucleus receive higher penalty.

    The loss uses soft sigmoid thresholding for nuclear mask creation and
    kornia's differentiable distance transform for gradient computation.

    Args:
        config: ExclusionConfig instance. Uses defaults if None.

    Example:
        >>> excl = ExclusionLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> loss, loss_dict = excl(pred)
        >>> loss.backward()  # Gradients flow to predictions
        >>> print(f"Exclusion loss: {loss.item():.4f}")

    Example with GT mask:
        >>> excl = ExclusionLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> target = torch.rand(4, 5, 256, 256)  # GT markers
        >>> loss, loss_dict = excl(pred, target)  # Uses GT LMNB1 for mask
    """

    def __init__(self, config: Optional[ExclusionConfig] = None):
        super().__init__()
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia required: pip install kornia>=0.7.0")

        self.config = config or ExclusionConfig()

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute exclusion loss.

        Args:
            pred: Predicted markers of shape (B, 5, H, W).
            target: Optional GT markers. If provided and use_gt_mask=True,
                    uses GT LMNB1 for nuclear mask creation.

        Returns:
            Tuple of (loss, loss_dict) where loss is scalar tensor and
            loss_dict contains 'loss_exclusion' key.
        """
        cfg = self.config

        # Determine which LMNB1 to use for mask
        if target is not None and cfg.use_gt_mask:
            lmnb1 = target[:, LMNB1_IDX, :, :]
        else:
            lmnb1 = pred[:, LMNB1_IDX, :, :]

        # Create soft nuclear mask
        nuclear_mask = create_nuclear_mask(
            lmnb1,
            use_otsu=cfg.use_otsu,
            fixed_threshold=cfg.fixed_threshold,
            temperature=cfg.temperature,
        )

        # Add channel dimension for distance_transform: (B, H, W) -> (B, 1, H, W)
        nuclear_mask_4d = nuclear_mask.unsqueeze(1)

        # Compute distance from nuclear boundary inward
        # kornia distance_transform computes distance TO nearest non-zero pixel
        # So distance_transform(1 - mask) gives us distance FROM the mask boundary
        # into the mask interior (pixels inside have high distance)
        outside_mask = 1.0 - nuclear_mask_4d
        dist_from_boundary = distance_transform(outside_mask)  # (B, 1, H, W)

        # Normalize to [0, 1] based on margin
        # Pixels deep inside nucleus get penalty weight ~1
        # Pixels near boundary get lower weight
        penalty_weight = torch.clamp(dist_from_boundary / cfg.margin_pixels, 0, 1)
        penalty_weight = penalty_weight.squeeze(1)  # (B, H, W)

        # Compute exclusion loss for each cytoplasmic marker
        exclusion_losses = []
        for idx in CYTOPLASMIC_MARKERS:
            cyto_marker = pred[:, idx, :, :]  # (B, H, W)

            # Penalize marker inside nuclear region, weighted by distance from edge
            # Higher penalty for signal deeper inside nucleus
            overlap = cyto_marker * nuclear_mask * penalty_weight
            loss_marker = overlap.mean()
            exclusion_losses.append(loss_marker)

        # Mean exclusion loss across all cytoplasmic markers
        loss = torch.stack(exclusion_losses).mean()

        loss_dict = {
            "loss_exclusion": loss.item(),
        }

        return loss, loss_dict


@dataclass
class ContainmentConfig:
    """Configuration for ContainmentLoss.

    Attributes:
        temperature: Sigmoid temperature for soft thresholding. Default 10.0.
        dilation_kernel: Kernel size for boundary dilation. Default 5.
        margin_pixels: Distance over which penalty ramps outside boundary. Default 10.
        use_otsu: Use adaptive Otsu threshold per batch. Default True.
        fixed_threshold: Nuclear threshold when use_otsu=False. Default 0.5.
        use_gt_mask: Use GT LMNB1 for mask if target provided. Default True.
    """
    temperature: float = 10.0
    dilation_kernel: int = 5
    margin_pixels: int = 10
    use_otsu: bool = True
    fixed_threshold: float = 0.5
    use_gt_mask: bool = True


class ContainmentLoss(nn.Module):
    """Differentiable containment loss for FBL marker.

    Penalizes FBL (nucleolar) signal appearing outside the nuclear boundary
    defined by LMNB1. Uses dilated boundary for tolerance and distance-weighted
    penalty so violations farther outside receive higher penalty.

    The loss uses soft sigmoid thresholding for nuclear mask creation,
    kornia morphological dilation for boundary expansion, and kornia's
    differentiable distance transform for gradient computation.

    Args:
        config: ContainmentConfig instance. Uses defaults if None.

    Example:
        >>> contain = ContainmentLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> loss, loss_dict = contain(pred)
        >>> loss.backward()  # Gradients flow to predictions
        >>> print(f"Containment loss: {loss.item():.4f}")

    Example with GT mask:
        >>> contain = ContainmentLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> target = torch.rand(4, 5, 256, 256)  # GT markers
        >>> loss, loss_dict = contain(pred, target)  # Uses GT LMNB1 for mask
    """

    def __init__(self, config: Optional[ContainmentConfig] = None):
        super().__init__()
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia required: pip install kornia>=0.7.0")

        self.config = config or ContainmentConfig()

        # Register dilation kernel as buffer (moves with module)
        k = self.config.dilation_kernel
        self.register_buffer("dilation_kernel", torch.ones(k, k))

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute containment loss.

        Args:
            pred: Predicted markers of shape (B, 5, H, W).
            target: Optional GT markers. If provided and use_gt_mask=True,
                    uses GT LMNB1 for nuclear mask creation.

        Returns:
            Tuple of (loss, loss_dict) where loss is scalar tensor and
            loss_dict contains 'loss_containment' key.
        """
        cfg = self.config

        # Determine which LMNB1 to use for mask
        if target is not None and cfg.use_gt_mask:
            lmnb1 = target[:, LMNB1_IDX, :, :]
        else:
            lmnb1 = pred[:, LMNB1_IDX, :, :]

        # Create soft nuclear mask
        nuclear_mask = create_nuclear_mask(
            lmnb1,
            use_otsu=cfg.use_otsu,
            fixed_threshold=cfg.fixed_threshold,
            temperature=cfg.temperature,
        )

        # Dilate mask to create tolerance region
        # Input to dilation must be (B, C, H, W)
        nuclear_mask_4d = nuclear_mask.unsqueeze(1)
        nuclear_dilated = morph.dilation(
            nuclear_mask_4d,
            self.dilation_kernel.to(nuclear_mask.device),
        )  # (B, 1, H, W)

        # Compute distance from dilated boundary outward
        # distance_transform(1 - mask) gives distance to nearest mask pixel
        # Pixels outside dilated region have high distance
        outside_mask = 1.0 - nuclear_dilated
        dist_outside = distance_transform(outside_mask)  # (B, 1, H, W)

        # Normalize to [0, 1] based on margin
        penalty_weight = torch.clamp(dist_outside / cfg.margin_pixels, 0, 1)
        penalty_weight = penalty_weight.squeeze(1)  # (B, H, W)
        nuclear_dilated = nuclear_dilated.squeeze(1)  # (B, H, W)

        # Get FBL signal
        fbl = pred[:, FBL_IDX, :, :]  # (B, H, W)

        # Penalize FBL outside dilated nuclear region
        # Higher penalty for signal farther from boundary
        outside_region = 1.0 - nuclear_dilated
        violation = fbl * outside_region * penalty_weight
        loss = violation.mean()

        loss_dict = {
            "loss_containment": loss.item(),
        }

        return loss, loss_dict


def manders_coefficients(
    channel_a: torch.Tensor,
    channel_b: torch.Tensor,
    threshold_a: float = 0.0,
    threshold_b: float = 0.0,
    temperature: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable Manders coefficients M1 and M2.

    M1 = fraction of A intensity in regions containing B
    M2 = fraction of B intensity in regions containing A

    Uses soft sigmoid thresholding for differentiability.

    Args:
        channel_a: First marker intensities (B, H, W).
        channel_b: Second marker intensities (B, H, W).
        threshold_a: Background threshold for channel A. Default 0.0.
        threshold_b: Background threshold for channel B. Default 0.0.
        temperature: Sigmoid temperature for soft thresholding. Default 10.0.

    Returns:
        Tuple of (M1, M2) as differentiable tensors.

    Example:
        >>> tomm20 = torch.rand(4, 256, 256)
        >>> sec61b = torch.rand(4, 256, 256)
        >>> m1, m2 = manders_coefficients(tomm20, sec61b)
        >>> # m1, m2 are in [0, 1] with gradient flow
    """
    # Soft co-occurrence weights using sigmoid
    weight_b = torch.sigmoid(temperature * (channel_b - threshold_b))
    weight_a = torch.sigmoid(temperature * (channel_a - threshold_a))

    # M1: fraction of A in regions with B
    a_in_b = (channel_a * weight_b).sum()
    a_total = channel_a.sum() + 1e-8
    m1 = a_in_b / a_total

    # M2: fraction of B in regions with A
    b_in_a = (channel_b * weight_a).sum()
    b_total = channel_b.sum() + 1e-8
    m2 = b_in_a / b_total

    return m1, m2


@dataclass
class ColocalizationConfig:
    """Configuration for ColocalizationLoss.

    Attributes:
        target_m1: Target Manders M1 coefficient (from GT audit). Default 0.5.
        target_m2: Target Manders M2 coefficient (from GT audit). Default 0.5.
        threshold_tomm20: Background threshold for TOMM20. Default 0.1.
        threshold_sec61b: Background threshold for SEC61B. Default 0.1.
        temperature: Sigmoid temperature for soft thresholding. Default 10.0.
        symmetric: Penalize deviation in both M1 and M2. Default True.
    """
    target_m1: float = 0.5
    target_m2: float = 0.5
    threshold_tomm20: float = 0.1
    threshold_sec61b: float = 0.1
    temperature: float = 10.0
    symmetric: bool = True


class ColocalizationLoss(nn.Module):
    """Differentiable colocalization loss for TOMM20-SEC61B.

    Encourages TOMM20 (mitochondria) and SEC61B (ER) to have appropriate
    spatial correlation reflecting mito-ER contact sites. Uses differentiable
    Manders coefficients M1 and M2.

    The loss penalizes deviation from target Manders values derived from
    ground truth data analysis. This encourages the model to produce
    realistic organelle spatial relationships.

    Args:
        config: ColocalizationConfig instance. Uses defaults if None.

    Example:
        >>> coloc = ColocalizationLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> loss, loss_dict = coloc(pred)
        >>> loss.backward()  # Gradients flow to predictions
        >>> print(f"Colocalization loss: {loss.item():.4f}")

    Example with custom targets:
        >>> config = ColocalizationConfig(target_m1=0.6, target_m2=0.4)
        >>> coloc = ColocalizationLoss(config)
        >>> loss, loss_dict = coloc(pred)
    """

    def __init__(self, config: Optional[ColocalizationConfig] = None):
        super().__init__()
        self.config = config or ColocalizationConfig()

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute colocalization loss.

        Args:
            pred: Predicted markers of shape (B, 5, H, W).
            target: Optional GT markers (unused, kept for API consistency).

        Returns:
            Tuple of (loss, loss_dict) where loss is scalar tensor and
            loss_dict contains 'loss_colocalization', 'manders_m1', 'manders_m2'.
        """
        cfg = self.config

        # Extract TOMM20 and SEC61B channels
        tomm20 = pred[:, TOMM20_IDX, :, :]  # (B, H, W)
        sec61b = pred[:, SEC61B_IDX, :, :]  # (B, H, W)

        # Compute differentiable Manders coefficients
        m1, m2 = manders_coefficients(
            tomm20,
            sec61b,
            threshold_a=cfg.threshold_tomm20,
            threshold_b=cfg.threshold_sec61b,
            temperature=cfg.temperature,
        )

        if cfg.symmetric:
            # Symmetric loss: penalize deviation in both directions
            loss = ((m1 - cfg.target_m1) ** 2 + (m2 - cfg.target_m2) ** 2) / 2
        else:
            # Asymmetric: encourage overlap without explicit targets
            loss = 1.0 - (m1 + m2) / 2

        loss_dict = {
            "loss_colocalization": loss.item(),
            "manders_m1": m1.item(),
            "manders_m2": m2.item(),
        }

        return loss, loss_dict


def compute_mitotic_weight(
    lmnb1: torch.Tensor,
    threshold: float = 0.3,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Compute per-sample weight based on nuclear envelope clarity.

    Mitotic cells have disrupted nuclear envelope (low/dispersed LMNB1).
    Returns lower weight for these samples.

    Detection heuristics:
    1. Low mean LMNB1 intensity
    2. Low fraction of high-signal pixels

    Args:
        lmnb1: LMNB1 channel (B, H, W).
        threshold: Below this mean intensity, consider mitotic.
        min_weight: Minimum constraint weight (never fully disable).

    Returns:
        Per-sample weights (B,) in [min_weight, 1.0].
    """
    batch_size = lmnb1.shape[0]
    weights = []

    for b in range(batch_size):
        sample = lmnb1[b]

        # Heuristic 1: Mean intensity
        mean_intensity = sample.mean()

        # Heuristic 2: Fraction of high-signal pixels
        high_signal_frac = (sample > threshold).float().mean()

        # If nuclear envelope is weak, reduce constraint weight
        if mean_intensity < threshold or high_signal_frac < 0.1:
            # Likely mitotic or poor nuclear signal
            weight = torch.tensor(min_weight, device=lmnb1.device, dtype=lmnb1.dtype)
        else:
            # Scale weight by nuclear clarity
            weight = torch.clamp(high_signal_frac * 2, min_weight, 1.0)

        weights.append(weight)

    return torch.stack(weights)


def should_skip_constraints(
    lmnb1: torch.Tensor,
    min_nuclear_fraction: float = 0.05,
) -> torch.Tensor:
    """Determine if constraints should be skipped for each sample.

    Skip if Otsu fails to find clear nuclear region.

    Args:
        lmnb1: LMNB1 channel (B, H, W).
        min_nuclear_fraction: Minimum fraction of nuclear pixels required.

    Returns:
        Boolean tensor (B,) - True means skip constraints for that sample.
    """
    try:
        from skimage.filters import threshold_otsu
    except ImportError:
        # If skimage not available, don't skip
        return torch.zeros(lmnb1.shape[0], dtype=torch.bool, device=lmnb1.device)

    skip = []
    for b in range(lmnb1.shape[0]):
        sample = lmnb1[b].detach().cpu().numpy()

        try:
            thresh = threshold_otsu(sample)
            nuclear_mask = sample > thresh
            nuclear_fraction = nuclear_mask.mean()

            # Skip if nuclear region is too small or too large
            should_skip = nuclear_fraction < min_nuclear_fraction or nuclear_fraction > 0.9
        except ValueError:
            # Otsu failed - skip constraints
            should_skip = True

        skip.append(bool(should_skip))

    return torch.tensor(skip, device=lmnb1.device, dtype=torch.bool)


@dataclass
class BiologicalConstraintConfig:
    """Configuration for BiologicalConstraintLoss wrapper.

    Attributes:
        weight_exclusion: Weight for exclusion constraint. Default 1.0.
        weight_containment: Weight for containment constraint. Default 1.0.
            Set to 0.0 to disable (prevents FBL from being affected by constraints).
        weight_colocalization: Weight for colocalization constraint. Default 1.0.
        enable_containment: Enable containment loss for FBL. Default True.
            Set to False to prevent biological constraints from affecting nuclear markers.
        threshold_temperature: Sigmoid temperature for soft thresholding. Default 10.0.
        use_otsu: Use adaptive Otsu threshold per batch. Default True.
        fixed_nuclear_threshold: Nuclear threshold when use_otsu=False. Default 0.5.
        margin_pixels: Distance over which penalty ramps. Default 10.
        dilation_kernel_size: Kernel size for containment boundary dilation. Default 5.
        target_m1: Target Manders M1 from GT audit (2026-01-19: 0.53). Default 0.53.
        target_m2: Target Manders M2 from GT audit (2026-01-19: 0.49). Default 0.49.
        handle_mitotic: Reduce weight for mitotic cells. Default True.
        min_mitotic_weight: Minimum weight for mitotic cells. Default 0.1.
        mitotic_threshold: Threshold for mitotic detection. Default 0.3.
        skip_if_no_nucleus: Skip constraints if Otsu fails. Default True.
        min_nuclear_fraction: Minimum nuclear fraction to apply constraints. Default 0.05.
        use_gt_mask: Use GT LMNB1 for nuclear mask during training. Default True.
    """
    # Weights for each constraint
    weight_exclusion: float = 1.0
    weight_containment: float = 1.0
    weight_colocalization: float = 1.0

    # Enable/disable containment (set False to not affect nuclear markers)
    enable_containment: bool = True

    # Shared parameters
    threshold_temperature: float = 10.0
    use_otsu: bool = True
    fixed_nuclear_threshold: float = 0.5
    margin_pixels: int = 10
    dilation_kernel_size: int = 5

    # Manders targets (from GT audit 2026-01-19)
    # M1 = fraction of TOMM20 in SEC61B regions = 0.5291
    # M2 = fraction of SEC61B in TOMM20 regions = 0.4916
    target_m1: float = 0.53
    target_m2: float = 0.49

    # Mitotic handling
    handle_mitotic: bool = True
    min_mitotic_weight: float = 0.1
    mitotic_threshold: float = 0.3

    # Skip constraints
    skip_if_no_nucleus: bool = True
    min_nuclear_fraction: float = 0.05

    # Mask source
    use_gt_mask: bool = True


class BiologicalConstraintLoss(nn.Module):
    """Combined biological constraint loss wrapper for training.

    Combines three differentiable constraint losses that encode known organelle
    spatial relationships into the training process:

    1. ExclusionLoss: Penalizes cytoplasmic markers inside nuclear region
    2. ContainmentLoss: Penalizes FBL outside nuclear boundary
    3. ColocalizationLoss: Encourages TOMM20-SEC61B spatial correlation

    Includes edge case handling:
    - Mitotic cell detection reduces constraint weight for weak nuclear envelope
    - Skip logic when Otsu fails to find clear nuclear region

    Args:
        config: BiologicalConstraintConfig instance. Uses defaults if None.

    Example:
        >>> from multiplex.training.biological_constraints import BiologicalConstraintLoss
        >>> import torch
        >>> loss_fn = BiologicalConstraintLoss()
        >>> pred = torch.rand(4, 5, 256, 256, requires_grad=True)
        >>> target = torch.rand(4, 5, 256, 256)
        >>> loss, loss_dict = loss_fn(pred, target)
        >>> loss.backward()
        >>> print(f"Total bio loss: {loss.item():.4f}")
        >>> print(f"Loss components: {loss_dict}")
    """

    def __init__(self, config: Optional[BiologicalConstraintConfig] = None):
        super().__init__()
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia required: pip install kornia>=0.7.0")

        self.config = config or BiologicalConstraintConfig()

        # Initialize individual loss components with matching parameters
        excl_config = ExclusionConfig(
            temperature=self.config.threshold_temperature,
            margin_pixels=self.config.margin_pixels,
            use_otsu=self.config.use_otsu,
            fixed_threshold=self.config.fixed_nuclear_threshold,
            use_gt_mask=self.config.use_gt_mask,
        )
        self.exclusion_loss = ExclusionLoss(excl_config)

        contain_config = ContainmentConfig(
            temperature=self.config.threshold_temperature,
            dilation_kernel=self.config.dilation_kernel_size,
            margin_pixels=self.config.margin_pixels,
            use_otsu=self.config.use_otsu,
            fixed_threshold=self.config.fixed_nuclear_threshold,
            use_gt_mask=self.config.use_gt_mask,
        )
        self.containment_loss = ContainmentLoss(contain_config)

        coloc_config = ColocalizationConfig(
            target_m1=self.config.target_m1,
            target_m2=self.config.target_m2,
            temperature=self.config.threshold_temperature,
        )
        self.colocalization_loss = ColocalizationLoss(coloc_config)

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined biological constraint loss.

        Args:
            pred: Predicted markers of shape (B, 5, H, W).
            target: Optional GT markers. If provided and use_gt_mask=True,
                    uses GT LMNB1 for nuclear mask creation.

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Weighted sum of all constraint losses
            - loss_dict: Dictionary with individual loss values:
                - loss_bio: Total biological constraint loss
                - loss_excl: Exclusion loss component
                - loss_contain: Containment loss component
                - loss_coloc: Colocalization loss component
        """
        cfg = self.config
        device = pred.device
        batch_size = pred.shape[0]

        # Determine which LMNB1 to use for mask and edge case detection
        if target is not None and cfg.use_gt_mask:
            lmnb1_for_mask = target[:, LMNB1_IDX, :, :]
        else:
            lmnb1_for_mask = pred[:, LMNB1_IDX, :, :]

        # Check if we should skip constraints for any samples
        if cfg.skip_if_no_nucleus:
            skip_mask = should_skip_constraints(
                lmnb1_for_mask,
                min_nuclear_fraction=cfg.min_nuclear_fraction,
            )
            if skip_mask.all():
                # All samples should skip - return zero loss
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                return zero, {
                    "loss_bio": 0.0,
                    "loss_excl": 0.0,
                    "loss_contain": 0.0,
                    "loss_coloc": 0.0,
                }
        else:
            skip_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Compute mitotic weights for per-sample constraint weighting
        if cfg.handle_mitotic:
            mitotic_weights = compute_mitotic_weight(
                lmnb1_for_mask,
                threshold=cfg.mitotic_threshold,
                min_weight=cfg.min_mitotic_weight,
            )
        else:
            mitotic_weights = torch.ones(batch_size, device=device)

        # Compute individual losses
        loss_excl, dict_excl = self.exclusion_loss(pred, target)

        # Containment loss affects FBL - skip if disabled
        if cfg.enable_containment and cfg.weight_containment > 0:
            loss_contain, dict_contain = self.containment_loss(pred, target)
        else:
            loss_contain = torch.tensor(0.0, device=device)
            dict_contain = {"loss_containment": 0.0}

        loss_coloc, dict_coloc = self.colocalization_loss(pred, target)

        # Apply mitotic weighting - average weight across batch
        # This gives reduced penalty for batches with mitotic cells
        mean_mitotic_weight = mitotic_weights[~skip_mask].mean() if not skip_mask.all() else 1.0

        # Apply weights and combine
        weighted_excl = cfg.weight_exclusion * loss_excl * mean_mitotic_weight
        weighted_contain = cfg.weight_containment * loss_contain * mean_mitotic_weight
        weighted_coloc = cfg.weight_colocalization * loss_coloc * mean_mitotic_weight

        total_loss = weighted_excl + weighted_contain + weighted_coloc

        loss_dict = {
            "loss_bio": total_loss.item(),
            "loss_excl": dict_excl["loss_exclusion"],
            "loss_contain": dict_contain["loss_containment"],
            "loss_coloc": dict_coloc["loss_colocalization"],
        }

        return total_loss, loss_dict


# Convenience exports
__all__ = [
    "ExclusionLoss",
    "ExclusionConfig",
    "ContainmentLoss",
    "ContainmentConfig",
    "ColocalizationLoss",
    "ColocalizationConfig",
    "BiologicalConstraintLoss",
    "BiologicalConstraintConfig",
    "manders_coefficients",
    "soft_threshold",
    "compute_otsu_threshold",
    "create_nuclear_mask",
    "compute_mitotic_weight",
    "should_skip_constraints",
    "LMNB1_IDX",
    "FBL_IDX",
    "TOMM20_IDX",
    "SEC61B_IDX",
    "ACTB_IDX",
    "CYTOPLASMIC_MARKERS",
]
