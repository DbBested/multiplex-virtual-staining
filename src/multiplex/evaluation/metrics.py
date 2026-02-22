"""Standard image quality metrics for virtual staining evaluation.

This module provides per-marker metric computation using PyIQA
for GPU-accelerated PSNR, SSIM, and LPIPS calculation.
"""

from typing import Dict, List, Optional
import torch
import numpy as np

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    pyiqa = None
    PYIQA_AVAILABLE = False


MARKERS = ["LMNB1", "FBL", "TOMM20", "SEC61B", "TUBA1B"]


class MarkerMetrics:
    """Compute per-marker image quality metrics with aggregation.

    Computes PSNR, SSIM, and LPIPS for each of 5 markers independently,
    then aggregates with mean and standard deviation.

    Args:
        device: Compute device (cuda or cpu).
        data_range: Expected data range (1.0 for [0,1] normalized).

    Example:
        >>> metrics = MarkerMetrics(device="cuda")
        >>> pred = torch.rand(4, 5, 256, 256, device="cuda")
        >>> target = torch.rand(4, 5, 256, 256, device="cuda")
        >>> results = metrics.compute(pred, target)
        >>> print(results["mean"]["psnr"])
    """

    def __init__(
        self,
        device: str = "cuda",
        data_range: float = 1.0,
    ):
        if not PYIQA_AVAILABLE:
            raise ImportError("pyiqa required: pip install pyiqa>=0.1.12")

        self.device = device
        self.data_range = data_range

        # Create metrics (pyiqa handles data_range internally for PSNR/SSIM)
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        # Use VGG backbone for LPIPS to match training loss
        self.lpips = pyiqa.create_metric('lpips-vgg', device=device)

    def _prepare_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare single-channel tensor for LPIPS (expects 3-ch, [0,1]).

        Note: PyIQA's LPIPS wrapper expects input in [0, 1] range and handles
        the normalization to [-1, 1] internally.

        Args:
            x: Tensor of shape (B, 1, H, W) in [0, 1] range.

        Returns:
            Tensor of shape (B, 3, H, W) in [0, 1] range.
        """
        # Expand single channel to 3 channels
        x_3ch = x.expand(-1, 3, -1, -1)
        return x_3ch

    @torch.no_grad()
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        markers: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per marker with aggregation.

        Args:
            pred: Predicted markers (B, C, H, W) in [0, 1].
            target: Ground truth markers (B, C, H, W) in [0, 1].
            markers: Optional list of marker names (defaults to MARKERS).

        Returns:
            Dictionary with per-marker results and aggregated mean/std.

        Example output:
            {
                "LMNB1": {"psnr": 28.5, "ssim": 0.85, "lpips": 0.12},
                "FBL": {"psnr": 27.2, "ssim": 0.82, "lpips": 0.15},
                ...
                "mean": {"psnr": 27.8, "ssim": 0.83, "lpips": 0.14},
                "std": {"psnr": 0.8, "ssim": 0.02, "lpips": 0.02}
            }
        """
        if markers is None:
            markers = MARKERS

        num_markers = pred.shape[1]
        assert num_markers == len(markers), f"Expected {len(markers)} markers, got {num_markers}"

        results = {}

        for i, marker in enumerate(markers):
            # Extract single marker channel
            p = pred[:, i:i+1, :, :]
            t = target[:, i:i+1, :, :]

            # PSNR and SSIM work on single channel
            # Expand to 3 channels for compatibility with pyiqa
            p_3ch = p.expand(-1, 3, -1, -1)
            t_3ch = t.expand(-1, 3, -1, -1)

            # Compute metrics
            psnr_val = self.psnr(p_3ch, t_3ch).mean().item()
            ssim_val = self.ssim(p_3ch, t_3ch).mean().item()

            # LPIPS needs [-1, 1] range
            p_lpips = self._prepare_for_lpips(p)
            t_lpips = self._prepare_for_lpips(t)
            lpips_val = self.lpips(p_lpips, t_lpips).mean().item()

            results[marker] = {
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
            }

        # Aggregate across markers
        metric_names = ["psnr", "ssim", "lpips"]
        results["mean"] = {
            m: float(np.mean([results[mk][m] for mk in markers]))
            for m in metric_names
        }
        results["std"] = {
            m: float(np.std([results[mk][m] for mk in markers]))
            for m in metric_names
        }

        return results


class BatchedMetricAccumulator:
    """Accumulate metrics over batches for full-dataset evaluation.

    Used for efficient evaluation over large test sets without
    storing all predictions in memory.

    Args:
        device: Compute device.

    Example:
        >>> accumulator = BatchedMetricAccumulator(device="cuda")
        >>> for batch in dataloader:
        ...     accumulator.update(pred, target)
        >>> results = accumulator.compute()
    """

    def __init__(self, device: str = "cuda"):
        self.metrics = MarkerMetrics(device=device)
        self.all_results: List[Dict] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add batch results to accumulator."""
        batch_results = self.metrics.compute(pred, target)
        self.all_results.append(batch_results)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregated metrics over all batches."""
        if not self.all_results:
            raise ValueError("No results accumulated")

        markers = MARKERS
        metric_names = ["psnr", "ssim", "lpips"]

        # Aggregate per-marker results
        results = {}
        for marker in markers:
            results[marker] = {
                m: float(np.mean([r[marker][m] for r in self.all_results]))
                for m in metric_names
            }

        # Compute mean and std across markers
        results["mean"] = {
            m: float(np.mean([results[mk][m] for mk in markers]))
            for m in metric_names
        }
        results["std"] = {
            m: float(np.std([results[mk][m] for mk in markers]))
            for m in metric_names
        }

        return results

    def reset(self) -> None:
        """Clear accumulated results."""
        self.all_results = []
