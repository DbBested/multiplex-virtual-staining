"""
Evaluation module for multiplex virtual staining.

Provides:
- Per-marker image quality metrics (PSNR, SSIM, LPIPS)
- FID computation with caching for baseline comparison
- Batched metric accumulation for large datasets
- Visual comparison figure generation
- LaTeX table export for paper
- Ablation study infrastructure
- Statistical testing for ablation comparisons
"""

from .metrics import (
    MarkerMetrics,
    BatchedMetricAccumulator,
    MARKERS,
)
from .fid import (
    FIDComputer,
    compute_fid_from_tensors,
    compute_fid_per_marker,
)
from .visualizer import (
    ComparisonVisualizer,
    create_comparison_grid,
    create_multi_marker_grid,
)
from .latex_export import (
    export_results_latex,
    export_per_marker_latex,
    export_ablation_latex,
    format_metric_cell,
)
from .ablations import (
    AblationConfig,
    ABLATION_CONFIGS,
    AblationRunner,
    run_single_ablation,
    generate_ablation_report,
    results_to_dataframe,
)
from .statistics import (
    compute_bootstrap_ci,
    cohens_d,
    compare_paired_bootstrap,
    format_comparison_table,
)

__all__ = [
    # Metrics
    "MarkerMetrics",
    "BatchedMetricAccumulator",
    "MARKERS",
    # FID
    "FIDComputer",
    "compute_fid_from_tensors",
    "compute_fid_per_marker",
    # Visualization
    "ComparisonVisualizer",
    "create_comparison_grid",
    "create_multi_marker_grid",
    # LaTeX export
    "export_results_latex",
    "export_per_marker_latex",
    "export_ablation_latex",
    "format_metric_cell",
    # Ablations
    "AblationConfig",
    "ABLATION_CONFIGS",
    "AblationRunner",
    "run_single_ablation",
    "generate_ablation_report",
    "results_to_dataframe",
    # Statistics
    "compute_bootstrap_ci",
    "cohens_d",
    "compare_paired_bootstrap",
    "format_comparison_table",
]
