"""Cell segmentation using Cellpose for per-cell analysis.

This module provides a Cellpose wrapper optimized for Allen Cell hiPSC images,
enabling instance segmentation for biological validation metrics.
"""

from typing import Tuple, Optional, Dict
import numpy as np

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    models = None
    CELLPOSE_AVAILABLE = False


class CellSegmenter:
    """Segment cells from fluorescence images using Cellpose.

    Optimized for Allen Cell hiPSC images where LMNB1 (nuclear envelope)
    serves as the nuclear proxy for segmentation.

    Attributes:
        model_type: Cellpose model ('cyto3' for cytoplasm, 'nuclei' for nuclei only)
        diameter: Expected cell diameter in pixels (60 for Allen Cell)
        gpu: Use GPU acceleration

    Example:
        >>> segmenter = CellSegmenter(model_type="cyto3", gpu=True)
        >>> mask = segmenter.segment(image)  # Returns integer-labeled mask
    """

    NUCLEAR_MARKER_IDX = 0  # LMNB1 is first marker

    def __init__(
        self,
        model_type: str = "cyto3",
        diameter: int = 60,
        gpu: bool = True,
    ):
        """Initialize CellSegmenter.

        Args:
            model_type: Cellpose model type. 'cyto3' for cytoplasm segmentation
                with nuclear channel, 'nuclei' for nuclei-only.
            diameter: Expected cell diameter in pixels.
            gpu: Whether to use GPU acceleration.
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError("cellpose required: pip install cellpose>=3.0")

        self.model = models.CellposeModel(model_type=model_type, gpu=gpu)
        self.diameter = diameter
        self.model_type = model_type

    def segment(
        self,
        image: np.ndarray,
        channels: list = None,
    ) -> np.ndarray:
        """Segment cells and return integer-labeled mask.

        Args:
            image: Input image (H, W) grayscale or (H, W, C)
            channels: Cellpose channel spec [cyto, nucleus]. Default [0,0] for grayscale.

        Returns:
            masks: Integer-labeled mask (H, W), 0=background, 1,2,...=cells
        """
        if channels is None:
            channels = [0, 0]

        masks, flows, styles = self.model.eval(
            image,
            diameter=self.diameter,
            channels=channels,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        return masks

    def segment_from_markers(
        self,
        marker_images: np.ndarray,
        nuclear_idx: int = None,
    ) -> np.ndarray:
        """Segment using nuclear marker as proxy.

        Args:
            marker_images: Marker stack (C, H, W) or (H, W, C)
            nuclear_idx: Index of nuclear marker. Default: 0 (LMNB1)

        Returns:
            masks: Integer-labeled segmentation mask
        """
        if nuclear_idx is None:
            nuclear_idx = self.NUCLEAR_MARKER_IDX

        # Handle both (C, H, W) and (H, W, C)
        if marker_images.ndim == 3:
            if marker_images.shape[0] <= 5:  # Likely (C, H, W)
                nuclear = marker_images[nuclear_idx]
            else:  # Likely (H, W, C)
                nuclear = marker_images[:, :, nuclear_idx]
        else:
            nuclear = marker_images

        return self.segment(nuclear)


def compute_instance_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Dice with instance matching.

    Matches predicted instances to GT instances using IoU threshold,
    then computes Dice for matched pairs.

    Args:
        pred_mask: Predicted instance segmentation (H, W)
        gt_mask: Ground truth instance segmentation (H, W)
        iou_threshold: Minimum IoU for instance matching

    Returns:
        dict with:
            - mean_dice: Average Dice over matched cells
            - matched_cells: Number of matched cells
            - total_pred: Number of predicted cells
            - total_gt: Number of GT cells
            - precision: matched / predicted
            - recall: matched / gt
    """
    pred_ids = np.unique(pred_mask[pred_mask > 0])
    gt_ids = np.unique(gt_mask[gt_mask > 0])

    matched_pred = set()
    matched_gt = set()
    dice_scores = []

    for pred_id in pred_ids:
        pred_region = (pred_mask == pred_id)
        best_iou = 0
        best_gt_id = None

        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            gt_region = (gt_mask == gt_id)
            intersection = np.sum(pred_region & gt_region)
            union = np.sum(pred_region | gt_region)
            iou = intersection / (union + 1e-8)

            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= iou_threshold and best_gt_id is not None:
            matched_pred.add(pred_id)
            matched_gt.add(best_gt_id)

            gt_region = (gt_mask == best_gt_id)
            intersection = np.sum(pred_region & gt_region)
            dice = 2 * intersection / (np.sum(pred_region) + np.sum(gt_region) + 1e-8)
            dice_scores.append(dice)

    return {
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "matched_cells": len(matched_pred),
        "total_pred": len(pred_ids),
        "total_gt": len(gt_ids),
        "precision": float(len(matched_pred) / (len(pred_ids) + 1e-8)),
        "recall": float(len(matched_gt) / (len(gt_ids) + 1e-8)),
    }


def compute_binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute binary Dice (foreground overlap only).

    Simple Dice coefficient between binary foreground masks, without
    instance-level matching.

    Args:
        pred_mask: Predicted segmentation mask (H, W), can be instance-labeled
        gt_mask: Ground truth segmentation mask (H, W), can be instance-labeled

    Returns:
        Dice coefficient in [0, 1]
    """
    pred_fg = pred_mask > 0
    gt_fg = gt_mask > 0
    intersection = np.sum(pred_fg & gt_fg)
    return float(2 * intersection / (np.sum(pred_fg) + np.sum(gt_fg) + 1e-8))
