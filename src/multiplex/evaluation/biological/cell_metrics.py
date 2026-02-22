"""
Per-cell metrics aggregator for biological validation.

Combines segmentation, intensity extraction, and co-expression analysis
into a unified validation pipeline.
"""

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch

from .segmentation import CellSegmenter, compute_instance_dice, compute_binary_dice
from .morphology import MorphologyExtractor, MARKERS
from .coexpression import CoExpressionAnalyzer


class PerCellMetrics:
    """Aggregator for per-cell biological validation metrics.

    Combines:
    - Cell segmentation (Cellpose)
    - Per-cell intensity extraction
    - Intensity correlation (pred vs GT)
    - Co-expression pattern comparison
    """

    def __init__(
        self,
        markers: List[str] = None,
        segmenter: CellSegmenter = None,
        gpu: bool = True,
    ):
        """Initialize metrics aggregator.

        Args:
            markers: List of marker names
            segmenter: Pre-initialized CellSegmenter, or create new one
            gpu: Use GPU for segmentation
        """
        self.markers = markers or MARKERS
        self.segmenter = segmenter or CellSegmenter(gpu=gpu)
        self.extractor = MorphologyExtractor(markers=self.markers)
        self.coexp_analyzer = CoExpressionAnalyzer(markers=self.markers)

    def compute(
        self,
        pred_markers: np.ndarray,
        gt_markers: np.ndarray,
        use_gt_mask: bool = True,
    ) -> dict:
        """Compute all per-cell metrics for a single image.

        Args:
            pred_markers: Predicted markers (C, H, W) in [0, 1]
            gt_markers: Ground truth markers (C, H, W) in [0, 1]
            use_gt_mask: If True, segment GT and apply to both (recommended for
                        intensity correlation - ensures same cells compared).
                        If False, segment both independently and compute
                        segmentation Dice (required for EVAL-06 segmentation
                        quality metric: >80% Dice overlap).

        Returns:
            dict with:
                - segmentation: Dice scores and cell counts
                - intensity_correlation: Per-marker Pearson r
                - coexpression: Co-expression comparison results
                - morphology: Feature correlation
        """
        # Ensure numpy arrays
        if isinstance(pred_markers, torch.Tensor):
            pred_markers = pred_markers.cpu().numpy()
        if isinstance(gt_markers, torch.Tensor):
            gt_markers = gt_markers.cpu().numpy()

        # Segment cells
        if use_gt_mask:
            # Use GT mask for both (ensures same cells compared)
            gt_mask = self.segmenter.segment_from_markers(gt_markers)
            pred_mask = gt_mask  # Same mask applied to predicted
            dice_result = {"mean_dice": 1.0, "note": "Using GT mask for both"}
        else:
            # Segment independently - REQUIRED for EVAL-06 segmentation quality
            gt_mask = self.segmenter.segment_from_markers(gt_markers)
            pred_mask = self.segmenter.segment_from_markers(pred_markers)
            dice_result = compute_instance_dice(pred_mask, gt_mask)

        # Extract per-cell intensities
        pred_intensities = self.extractor.extract_intensities(gt_mask, pred_markers)
        gt_intensities = self.extractor.extract_intensities(gt_mask, gt_markers)

        # Compute per-cell intensity correlations
        intensity_corr = compute_per_cell_correlation(pred_intensities, gt_intensities, self.markers)

        # Compare co-expression patterns
        coexp_result = self.coexp_analyzer.compare_coexpression(pred_intensities, gt_intensities)

        # Extract morphology features and compare
        pred_features = self.extractor.extract_features(gt_mask, pred_markers)
        gt_features = self.extractor.extract_features(gt_mask, gt_markers)

        return {
            "segmentation": dice_result,
            "n_cells": len(pred_intensities),
            "intensity_correlation": intensity_corr,
            "coexpression": coexp_result,
            "pred_intensities": pred_intensities,
            "gt_intensities": gt_intensities,
            "morphology": {
                "pred": pred_features,
                "gt": gt_features,
            },
        }

    def compute_batch(
        self,
        pred_batch: np.ndarray,
        gt_batch: np.ndarray,
        use_gt_mask: bool = True,
    ) -> dict:
        """Compute metrics over a batch and aggregate.

        Args:
            pred_batch: Predicted markers (B, C, H, W)
            gt_batch: Ground truth markers (B, C, H, W)
            use_gt_mask: If True, use GT masks for intensity comparison.
                        If False, segment independently for Dice computation.

        Returns:
            Aggregated metrics across batch
        """
        all_pred_intensities = []
        all_gt_intensities = []
        all_dice = []
        total_cells = 0

        for i in range(len(pred_batch)):
            result = self.compute(pred_batch[i], gt_batch[i], use_gt_mask=use_gt_mask)
            all_pred_intensities.append(result["pred_intensities"])
            all_gt_intensities.append(result["gt_intensities"])
            all_dice.append(result["segmentation"].get("mean_dice", 1.0))
            total_cells += result["n_cells"]

        # Concatenate all cell intensities
        pred_combined = pd.concat(all_pred_intensities, ignore_index=True)
        gt_combined = pd.concat(all_gt_intensities, ignore_index=True)

        # Compute overall correlations
        intensity_corr = compute_per_cell_correlation(pred_combined, gt_combined, self.markers)
        coexp_result = self.coexp_analyzer.compare_coexpression(pred_combined, gt_combined)

        return {
            "n_images": len(pred_batch),
            "n_cells": total_cells,
            "mean_dice": np.mean(all_dice),
            "intensity_correlation": intensity_corr,
            "coexpression": coexp_result,
        }


def compute_per_cell_correlation(
    pred_intensities: pd.DataFrame,
    gt_intensities: pd.DataFrame,
    markers: List[str] = None,
) -> dict:
    """Compute per-cell Pearson correlation for each marker.

    Args:
        pred_intensities: DataFrame with columns [cell_id, marker1, marker2, ...]
        gt_intensities: DataFrame with same structure
        markers: List of marker names to compute correlation for

    Returns:
        dict with per-marker correlation and aggregate statistics
    """
    markers = markers or MARKERS
    results = {}

    for marker in markers:
        if marker in pred_intensities.columns and marker in gt_intensities.columns:
            # Cast to float64 to handle np.isnan on int arrays from regionprops
            pred_vals = pred_intensities[marker].values.astype(np.float64)
            gt_vals = gt_intensities[marker].values.astype(np.float64)

            # Filter valid values
            valid = ~(np.isnan(pred_vals) | np.isnan(gt_vals))
            if np.sum(valid) > 2:
                r, p = pearsonr(pred_vals[valid], gt_vals[valid])
            else:
                r, p = np.nan, np.nan

            results[marker] = {
                "pearson_r": r,
                "p_value": p,
                "n_cells": int(np.sum(valid)),
            }

    # Aggregate
    r_values = [results[m]["pearson_r"] for m in markers if m in results and not np.isnan(results[m]["pearson_r"])]
    if r_values:
        results["mean"] = {
            "pearson_r": np.mean(r_values),
            "pearson_r_std": np.std(r_values),
            "n_markers": len(r_values),
        }
    else:
        results["mean"] = {"pearson_r": np.nan, "pearson_r_std": np.nan, "n_markers": 0}

    return results
