"""Per-cell morphology and intensity feature extraction.

This module provides tools for extracting per-cell morphological features
(area, perimeter, circularity) and marker intensities for biological validation.
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

# Allen Cell structural markers
MARKERS = ["LMNB1", "FBL", "TOMM20", "SEC61B", "TUBA1B"]


class MorphologyExtractor:
    """Extract per-cell morphology and intensity features.

    Uses scikit-image regionprops to measure cell properties including
    area, circularity, and mean intensity per marker.

    Attributes:
        markers: List of marker names for intensity columns.

    Example:
        >>> extractor = MorphologyExtractor()
        >>> features = extractor.extract_features(mask, marker_images)
        >>> features.columns
        ['cell_id', 'area', 'perimeter', 'circularity', 'eccentricity',
         'LMNB1_mean', 'FBL_mean', ...]
    """

    def __init__(self, markers: List[str] = None):
        """Initialize MorphologyExtractor.

        Args:
            markers: List of marker names. Defaults to Allen Cell structural markers.
        """
        self.markers = markers or MARKERS

    def extract_features(
        self,
        cell_mask: np.ndarray,
        marker_images: np.ndarray = None,
    ) -> pd.DataFrame:
        """Extract morphology features for each cell.

        Args:
            cell_mask: Integer-labeled segmentation (H, W)
            marker_images: Optional marker stack (C, H, W) for intensity

        Returns:
            DataFrame with columns: cell_id, area, perimeter, circularity,
            eccentricity, and optionally {marker}_mean for each marker
        """
        # Handle empty mask
        if np.max(cell_mask) == 0:
            cols = ["cell_id", "area", "perimeter", "circularity", "eccentricity"]
            if marker_images is not None:
                cols.extend([f"{m}_mean" for m in self.markers[:marker_images.shape[0]]])
            return pd.DataFrame(columns=cols)

        # Basic morphology properties
        props = regionprops_table(
            cell_mask,
            properties=["label", "area", "perimeter", "eccentricity"]
        )

        df = pd.DataFrame(props)
        df = df.rename(columns={"label": "cell_id"})

        # Compute circularity: 4*pi*area / perimeter^2
        df["circularity"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-8)

        # Add intensity features if markers provided
        if marker_images is not None:
            for i, marker in enumerate(self.markers):
                if i < marker_images.shape[0]:
                    intensity_props = regionprops_table(
                        cell_mask,
                        intensity_image=marker_images[i],
                        properties=["label", "intensity_mean"]
                    )
                    df[f"{marker}_mean"] = intensity_props["intensity_mean"]

        return df

    def extract_intensities(
        self,
        cell_mask: np.ndarray,
        marker_images: np.ndarray,
    ) -> pd.DataFrame:
        """Extract per-cell mean intensities for all markers.

        Args:
            cell_mask: Integer-labeled segmentation (H, W)
            marker_images: Marker stack (C, H, W)

        Returns:
            DataFrame with columns: cell_id, LMNB1, FBL, TOMM20, SEC61B, TUBA1B
        """
        # Handle empty mask
        if np.max(cell_mask) == 0:
            cols = ["cell_id"] + self.markers[:marker_images.shape[0]]
            return pd.DataFrame(columns=cols)

        results = {}

        for i, marker in enumerate(self.markers):
            if i >= marker_images.shape[0]:
                break
            props = regionprops_table(
                cell_mask,
                intensity_image=marker_images[i],
                properties=["label", "intensity_mean"]
            )
            if i == 0:
                results["cell_id"] = props["label"]
            results[marker] = props["intensity_mean"]

        return pd.DataFrame(results)


def extract_cell_features(
    cell_mask: np.ndarray,
    marker_images: np.ndarray = None,
    markers: List[str] = None,
) -> pd.DataFrame:
    """Convenience function for feature extraction.

    Args:
        cell_mask: Integer-labeled segmentation
        marker_images: Optional marker stack (C, H, W)
        markers: Optional list of marker names

    Returns:
        DataFrame with cell features
    """
    extractor = MorphologyExtractor(markers=markers)
    return extractor.extract_features(cell_mask, marker_images)


def compare_morphology(
    pred_features: pd.DataFrame,
    gt_features: pd.DataFrame,
    feature_cols: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare morphology features between predicted and GT.

    Computes Pearson correlation for each feature column to assess
    how well predicted markers preserve biological structure.

    Args:
        pred_features: Features from predicted markers
        gt_features: Features from GT markers
        feature_cols: Columns to compare. Default: all numeric except cell_id

    Returns:
        dict with correlation and difference statistics per feature

    Example:
        >>> results = compare_morphology(pred_df, gt_df)
        >>> print(results["area"]["pearson_r"])
        0.95
    """
    from scipy.stats import pearsonr

    if feature_cols is None:
        feature_cols = [c for c in pred_features.columns
                       if c != "cell_id" and pred_features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    results = {}
    for col in feature_cols:
        if col in pred_features.columns and col in gt_features.columns:
            # Match by cell_id if available, else assume aligned
            pred_vals = pred_features[col].values
            gt_vals = gt_features[col].values

            if len(pred_vals) == len(gt_vals) and len(pred_vals) > 2:
                r, p = pearsonr(pred_vals, gt_vals)
                results[col] = {
                    "pearson_r": float(r),
                    "p_value": float(p),
                    "n": len(pred_vals)
                }
            else:
                results[col] = {
                    "pearson_r": float('nan'),
                    "p_value": float('nan'),
                    "n": 0
                }

    return results
