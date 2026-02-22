"""Biological validation metrics for multiplex virtual staining.

This module provides:
- Cell segmentation using Cellpose
- Instance Dice coefficient computation
- Per-cell morphology feature extraction
- Marker co-expression analysis
- Per-cell metrics aggregation
- Validation report generation
- BioValidity composite metric for biological plausibility
"""

from .segmentation import (
    CellSegmenter,
    compute_instance_dice,
    compute_binary_dice,
    CELLPOSE_AVAILABLE,
)
from .morphology import (
    MorphologyExtractor,
    extract_cell_features,
    compare_morphology,
    MARKERS,
)
from .coexpression import (
    CoExpressionAnalyzer,
    compute_coexpression_matrix,
    EXPECTED_CORRELATIONS,
)
from .cell_metrics import (
    PerCellMetrics,
    compute_per_cell_correlation,
)
from .validation_report import (
    generate_validation_report,
    plot_coexpression_heatmap,
    plot_intensity_correlation,
)
from .biovalidity import (
    BioValidityMetric,
    BioValidityConfig,
    log_biovalidity_to_wandb,
)

__all__ = [
    # Segmentation
    "CellSegmenter",
    "compute_instance_dice",
    "compute_binary_dice",
    "CELLPOSE_AVAILABLE",
    # Morphology
    "MorphologyExtractor",
    "extract_cell_features",
    "compare_morphology",
    "MARKERS",
    # Co-expression
    "CoExpressionAnalyzer",
    "compute_coexpression_matrix",
    "EXPECTED_CORRELATIONS",
    # Cell metrics
    "PerCellMetrics",
    "compute_per_cell_correlation",
    # Report
    "generate_validation_report",
    "plot_coexpression_heatmap",
    "plot_intensity_correlation",
    # BioValidity
    "BioValidityMetric",
    "BioValidityConfig",
    "log_biovalidity_to_wandb",
]
