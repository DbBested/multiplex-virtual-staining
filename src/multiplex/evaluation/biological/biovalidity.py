"""BioValidity composite metric for biological plausibility evaluation.

This module provides a composite metric that evaluates biological plausibility
of virtual staining predictions based on three sub-metrics:

1. Exclusion score: Cytoplasmic markers (TOMM20, SEC61B, ACTB) should avoid
   the nuclear interior.
2. Containment score: FBL (nucleoli) should be contained within the LMNB1
   (nuclear envelope) boundary.
3. Colocalization score: TOMM20 and SEC61B should show positive spatial
   correlation (mito-ER contacts).

The composite BioValidity score is a weighted average of these three sub-metrics,
providing a single measure of how well predictions respect known biological
spatial relationships.

Example:
    >>> from src.multiplex.evaluation.biological import BioValidityMetric
    >>> metric = BioValidityMetric()
    >>> pred = torch.rand(4, 5, 256, 256, device="cuda")
    >>> results = metric.compute(pred)
    >>> print(results["biovalidity"])  # Composite score in [0, 1]
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import numpy as np

try:
    from kornia import morphology as morph
    KORNIA_AVAILABLE = True
except ImportError:
    morph = None
    KORNIA_AVAILABLE = False


@dataclass
class BioValidityConfig:
    """Configuration for BioValidity metric computation.

    Attributes:
        nuclear_threshold: Threshold for LMNB1 binarization (default: 0.5).
        marker_threshold: General threshold for marker binarization (default: 0.3).
        nuclear_dilation_kernel: Kernel size for nuclear boundary dilation,
            providing tolerance for containment scoring (default: 5).
        weight_exclusion: Weight for exclusion score in composite (default: 0.4).
        weight_containment: Weight for containment score in composite (default: 0.3).
        weight_colocalization: Weight for colocalization in composite (default: 0.3).
        device: Compute device (default: "cuda").
    """
    nuclear_threshold: float = 0.5
    marker_threshold: float = 0.3
    nuclear_dilation_kernel: int = 5
    weight_exclusion: float = 0.4
    weight_containment: float = 0.3
    weight_colocalization: float = 0.3
    device: str = "cuda"


class BioValidityMetric:
    """Composite biological validity metric.

    Computes three sub-metrics that evaluate biological plausibility:
    1. Exclusion score: cytoplasmic markers outside nucleus
    2. Containment score: FBL inside LMNB1 boundary
    3. Colocalization score: TOMM20-SEC61B spatial correlation

    The composite BioValidity score is a weighted average of the three.

    Args:
        config: Optional configuration. Defaults to BioValidityConfig().

    Example:
        >>> metric = BioValidityMetric()
        >>> pred = torch.rand(2, 5, 128, 128)
        >>> results = metric.compute(pred)
        >>> print(f"BioValidity: {results['biovalidity']:.3f}")
    """

    # Marker channel order
    MARKERS = ["LMNB1", "FBL", "TOMM20", "SEC61B", "ACTB"]
    NUCLEAR_IDX = 0  # LMNB1
    FBL_IDX = 1
    TOMM20_IDX = 2
    SEC61B_IDX = 3
    ACTB_IDX = 4

    def __init__(self, config: Optional[BioValidityConfig] = None):
        """Initialize BioValidityMetric.

        Args:
            config: Configuration for metric computation. If None, uses defaults.
        """
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia required: pip install kornia>=0.7.0")

        self.config = config or BioValidityConfig()
        self.device = self.config.device

        # Create morphological dilation kernel
        k = self.config.nuclear_dilation_kernel
        self.dilation_kernel = torch.ones(k, k, device=self.device)

    @torch.no_grad()
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor = None,
    ) -> Dict[str, float]:
        """Compute all sub-metrics and composite BioValidity score.

        Args:
            pred: Predicted marker images of shape (B, 5, H, W) in [0, 1] range.
                  Channels: [LMNB1, FBL, TOMM20, SEC61B, ACTB].
            target: Optional ground truth for reference comparison.
                    Currently unused but reserved for future extensions.

        Returns:
            Dictionary containing:
                - exclusion_score: Score in [0, 1], higher = better nuclear avoidance
                - containment_score: Score in [0, 1], higher = better FBL containment
                - colocalization_r: Normalized correlation in [0, 1], higher = stronger
                - biovalidity: Weighted composite score in [0, 1]
        """
        # Move to device if needed
        if pred.device.type != self.device and self.device != "cpu":
            pred = pred.to(self.device)
        elif self.device == "cpu" and pred.device.type != "cpu":
            pred = pred.cpu()

        # Ensure kernel is on same device as input
        if self.dilation_kernel.device != pred.device:
            self.dilation_kernel = self.dilation_kernel.to(pred.device)

        results = {}

        # 1. Exclusion score: cytoplasmic markers avoiding nucleus
        results["exclusion_score"] = self._compute_exclusion(pred)

        # 2. Containment score: FBL inside nuclear boundary
        results["containment_score"] = self._compute_containment(pred)

        # 3. Colocalization score: TOMM20-SEC61B correlation
        results["colocalization_r"] = self._compute_colocalization(pred)

        # Composite BioValidity score (weighted average)
        cfg = self.config
        results["biovalidity"] = (
            cfg.weight_exclusion * results["exclusion_score"] +
            cfg.weight_containment * results["containment_score"] +
            cfg.weight_colocalization * results["colocalization_r"]
        )

        return results

    def _compute_exclusion(self, pred: torch.Tensor) -> float:
        """Compute exclusion score: 1 - (cytoplasmic signal in nucleus).

        Measures how well cytoplasmic markers (TOMM20, SEC61B, ACTB) avoid
        the nuclear interior region defined by LMNB1.

        Higher score = better exclusion = more biological plausibility.

        Args:
            pred: Predictions of shape (B, 5, H, W).

        Returns:
            Mean exclusion score across cytoplasmic markers in [0, 1].
        """
        # Get nuclear mask from LMNB1
        lmnb1 = pred[:, self.NUCLEAR_IDX, :, :]  # (B, H, W)
        nuclear_mask = (lmnb1 > self.config.nuclear_threshold).float()

        # Fill holes to get solid nuclear interior
        nuclear_interior = self._fill_nuclear_holes(nuclear_mask)

        # Cytoplasmic marker indices
        cyto_indices = [self.TOMM20_IDX, self.SEC61B_IDX, self.ACTB_IDX]

        exclusion_scores = []
        for idx in cyto_indices:
            cyto_marker = pred[:, idx, :, :]  # (B, H, W)

            # Measure overlap: cytoplasmic signal inside nucleus
            overlap = (cyto_marker * nuclear_interior).sum()
            total_cyto = cyto_marker.sum() + 1e-8

            # Exclusion = 1 - overlap_ratio
            overlap_ratio = (overlap / total_cyto).item()
            exclusion_scores.append(1.0 - overlap_ratio)

        return float(np.mean(exclusion_scores))

    def _compute_containment(self, pred: torch.Tensor) -> float:
        """Compute containment score: fraction of FBL inside nuclear boundary.

        Measures how well FBL (nucleoli) is contained within the LMNB1
        (nuclear envelope) boundary. A small dilation is applied to the
        nuclear boundary to provide tolerance for prediction uncertainty.

        Higher score = better containment = more biological plausibility.

        Args:
            pred: Predictions of shape (B, 5, H, W).

        Returns:
            Containment score in [0, 1].
        """
        # Get nuclear mask from LMNB1
        lmnb1 = pred[:, self.NUCLEAR_IDX, :, :]  # (B, H, W)
        nuclear_mask = (lmnb1 > self.config.nuclear_threshold).float()

        # Fill holes to get solid nuclear interior
        nuclear_interior = self._fill_nuclear_holes(nuclear_mask)

        # Dilate to create tolerance region around boundary
        nuclear_dilated = morph.dilation(
            nuclear_interior.unsqueeze(1),  # (B, 1, H, W)
            self.dilation_kernel
        ).squeeze(1)  # (B, H, W)

        # Get FBL signal
        fbl = pred[:, self.FBL_IDX, :, :]  # (B, H, W)

        # Containment = FBL inside dilated nuclear region / total FBL
        fbl_inside = (fbl * nuclear_dilated).sum()
        fbl_total = fbl.sum() + 1e-8

        return float((fbl_inside / fbl_total).item())

    def _compute_colocalization(self, pred: torch.Tensor) -> float:
        """Compute colocalization correlation between TOMM20 and SEC61B.

        Measures the Pearson correlation coefficient between TOMM20
        (mitochondria) and SEC61B (ER) intensity patterns, reflecting
        mito-ER contact site proximity.

        The raw correlation r in [-1, 1] is normalized to [0, 1] for
        the composite score.

        Args:
            pred: Predictions of shape (B, 5, H, W).

        Returns:
            Normalized colocalization score in [0, 1].
            0.5 = no correlation, 1.0 = perfect positive correlation.
        """
        # Extract TOMM20 and SEC61B channels and flatten
        tomm20 = pred[:, self.TOMM20_IDX, :, :].flatten()
        sec61b = pred[:, self.SEC61B_IDX, :, :].flatten()

        # Compute Pearson correlation
        tomm20_centered = tomm20 - tomm20.mean()
        sec61b_centered = sec61b - sec61b.mean()

        numerator = (tomm20_centered * sec61b_centered).sum()
        denominator = (
            torch.sqrt((tomm20_centered ** 2).sum()) *
            torch.sqrt((sec61b_centered ** 2).sum())
        ) + 1e-8

        r = (numerator / denominator).item()

        # Normalize to [0, 1] range for composite score
        # Map [-1, 1] to [0, 1]: 0.5 = no correlation, 1.0 = perfect positive
        return float((r + 1) / 2)

    def _fill_nuclear_holes(self, nuclear_mask: torch.Tensor) -> torch.Tensor:
        """Fill holes in nuclear mask to get solid interior.

        Uses scipy.ndimage.binary_fill_holes on CPU for robust hole filling,
        then returns the result to the appropriate device.

        Args:
            nuclear_mask: Binary mask of shape (B, H, W).

        Returns:
            Filled mask of shape (B, H, W) on the input device.
        """
        from scipy.ndimage import binary_fill_holes

        device = nuclear_mask.device

        # Process each sample
        filled = []
        for i in range(nuclear_mask.shape[0]):
            mask_np = nuclear_mask[i].cpu().numpy()
            filled_np = binary_fill_holes(mask_np > 0.5)
            filled.append(torch.from_numpy(filled_np.astype(np.float32)))

        return torch.stack(filled).to(device)


def log_biovalidity_to_wandb(
    results: Dict[str, float],
    step: int,
    prefix: str = "biovalidity",
) -> None:
    """Log BioValidity sub-metrics to W&B.

    Each sub-metric is logged independently for detailed analysis:
    - {prefix}/exclusion_score
    - {prefix}/containment_score
    - {prefix}/colocalization_r
    - {prefix}/composite

    This function is safe to call even if W&B is not initialized - it will
    silently return without error.

    Args:
        results: Output dictionary from BioValidityMetric.compute().
        step: Global step for logging.
        prefix: Metric prefix for W&B (e.g., "biovalidity", "biovalidity_test").

    Example:
        >>> metric = BioValidityMetric()
        >>> results = metric.compute(pred)
        >>> log_biovalidity_to_wandb(results, step=100)
    """
    try:
        import wandb

        if wandb.run is None:
            return

        metrics = {
            f"{prefix}/exclusion_score": results["exclusion_score"],
            f"{prefix}/containment_score": results["containment_score"],
            f"{prefix}/colocalization_r": results["colocalization_r"],
            f"{prefix}/composite": results["biovalidity"],
        }

        wandb.log(metrics, step=step)
    except ImportError:
        pass  # wandb not installed
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to log BioValidity to W&B: {e}")
