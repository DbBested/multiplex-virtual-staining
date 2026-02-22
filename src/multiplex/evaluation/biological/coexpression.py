"""
Co-expression analysis for structural marker validation.

Computes pairwise marker correlations to validate that predicted markers
preserve biological spatial relationships (e.g., LMNB1-FBL nuclear proximity,
TOMM20-SEC61B mito-ER contacts).
"""

from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

# Biologically meaningful marker pairs for structural proteins
EXPECTED_CORRELATIONS = {
    # Nuclear markers should positively correlate
    ("LMNB1", "FBL"): {"expected": "positive", "description": "Nuclear envelope - nucleolus"},
    # Cytoplasmic organelles may correlate (mito-ER contacts)
    ("TOMM20", "SEC61B"): {"expected": "positive", "description": "Mitochondria - ER contacts"},
    # Nuclear vs cytoplasmic should be weakly correlated
    ("LMNB1", "TOMM20"): {"expected": "weak", "description": "Nuclear vs mitochondria"},
    ("FBL", "TOMM20"): {"expected": "weak", "description": "Nucleolus vs mitochondria"},
}


class CoExpressionAnalyzer:
    """Analyze co-expression patterns between structural markers.

    Computes pairwise Pearson correlations across cells and compares
    patterns between predicted and real marker images.
    """

    def __init__(self, markers: List[str] = None):
        """Initialize analyzer.

        Args:
            markers: List of marker names. Default: Allen Cell structural markers.
        """
        self.markers = markers or ["LMNB1", "FBL", "TOMM20", "SEC61B", "TUBA1B"]

    def compute_coexpression_matrix(
        self,
        cell_intensities: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pairwise Pearson correlations across all markers.

        Args:
            cell_intensities: DataFrame with columns [cell_id, LMNB1, FBL, ...]

        Returns:
            Tuple of (correlation_matrix, pvalue_matrix) as DataFrames
        """
        # Get marker columns from DataFrame
        marker_cols = [c for c in self.markers if c in cell_intensities.columns]
        n = len(marker_cols)

        corr_matrix = np.zeros((n, n))
        pval_matrix = np.zeros((n, n))

        for i, m1 in enumerate(marker_cols):
            for j, m2 in enumerate(marker_cols):
                if i <= j:
                    # Cast to float64 to handle np.isnan on int arrays from regionprops
                    v1 = cell_intensities[m1].values.astype(np.float64)
                    v2 = cell_intensities[m2].values.astype(np.float64)

                    # Handle NaN values
                    valid = ~(np.isnan(v1) | np.isnan(v2))
                    if np.sum(valid) > 2:
                        r, p = pearsonr(v1[valid], v2[valid])
                    else:
                        r, p = np.nan, np.nan

                    corr_matrix[i, j] = r
                    corr_matrix[j, i] = r
                    pval_matrix[i, j] = p
                    pval_matrix[j, i] = p

        return (
            pd.DataFrame(corr_matrix, index=marker_cols, columns=marker_cols),
            pd.DataFrame(pval_matrix, index=marker_cols, columns=marker_cols),
        )

    def compare_coexpression(
        self,
        pred_intensities: pd.DataFrame,
        real_intensities: pd.DataFrame,
    ) -> dict:
        """Compare co-expression patterns between predicted and real.

        Args:
            pred_intensities: Per-cell intensities from predicted markers
            real_intensities: Per-cell intensities from real markers

        Returns:
            dict with:
                - pred_coexpression: Correlation matrix for predicted
                - real_coexpression: Correlation matrix for real
                - pattern_correlation: Pearson r between flattened matrices
                - pair_comparisons: Per-pair correlation comparison
        """
        pred_corr, pred_pval = self.compute_coexpression_matrix(pred_intensities)
        real_corr, real_pval = self.compute_coexpression_matrix(real_intensities)

        # Compare overall patterns: flatten upper triangle and correlate
        n = len(pred_corr)
        triu_idx = np.triu_indices(n, k=1)
        pred_flat = pred_corr.values[triu_idx]
        real_flat = real_corr.values[triu_idx]

        # Remove NaN pairs
        valid = ~(np.isnan(pred_flat) | np.isnan(real_flat))
        if np.sum(valid) > 2:
            pattern_corr, pattern_pval = pearsonr(pred_flat[valid], real_flat[valid])
        else:
            pattern_corr, pattern_pval = np.nan, np.nan

        # Compare biologically meaningful pairs
        pair_comparisons = {}
        for (m1, m2), info in EXPECTED_CORRELATIONS.items():
            if m1 in pred_corr.index and m2 in pred_corr.columns:
                pair_comparisons[(m1, m2)] = {
                    "pred_r": pred_corr.loc[m1, m2],
                    "real_r": real_corr.loc[m1, m2],
                    "expected": info["expected"],
                    "description": info["description"],
                    "preserved": self._check_pattern_preserved(
                        pred_corr.loc[m1, m2], real_corr.loc[m1, m2], info["expected"]
                    ),
                }

        return {
            "pred_coexpression": pred_corr,
            "real_coexpression": real_corr,
            "pred_pvalues": pred_pval,
            "real_pvalues": real_pval,
            "pattern_correlation": pattern_corr,
            "pattern_pvalue": pattern_pval,
            "pair_comparisons": pair_comparisons,
        }

    def _check_pattern_preserved(
        self, pred_r: float, real_r: float, expected: str
    ) -> bool:
        """Check if expected correlation pattern is preserved."""
        if np.isnan(pred_r) or np.isnan(real_r):
            return False

        if expected == "positive":
            # Both should be positive and similar sign
            return pred_r > 0.2 and real_r > 0.2
        elif expected == "weak":
            # Both should be near zero
            return abs(pred_r) < 0.5 and abs(real_r) < 0.5
        else:
            return True


def compute_coexpression_matrix(
    cell_intensities: pd.DataFrame,
    markers: List[str] = None,
) -> pd.DataFrame:
    """Convenience function for computing co-expression matrix.

    Args:
        cell_intensities: DataFrame with per-cell marker intensities
        markers: List of marker names to include

    Returns:
        Correlation matrix as DataFrame
    """
    analyzer = CoExpressionAnalyzer(markers=markers)
    corr, _ = analyzer.compute_coexpression_matrix(cell_intensities)
    return corr
