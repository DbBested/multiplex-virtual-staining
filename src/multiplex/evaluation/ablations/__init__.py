"""
Ablation study infrastructure for systematic component evaluation.

This module provides:
- AblationConfig: Configuration dataclass for ablation experiments
- ABLATION_CONFIGS: 7 standard ablation configurations
- AblationRunner: Training and evaluation for ablation experiments
- Report generation functions for LaTeX output
"""

from .ablation_config import (
    AblationConfig,
    ABLATION_CONFIGS,
    get_ablation_config,
)
from .ablation_runner import (
    AblationRunner,
    run_single_ablation,
)
from .ablation_report import (
    generate_ablation_report,
    results_to_dataframe,
    load_ablation_results,
    format_ablation_summary,
)

__all__ = [
    # Config
    "AblationConfig",
    "ABLATION_CONFIGS",
    "get_ablation_config",
    # Runner
    "AblationRunner",
    "run_single_ablation",
    # Report
    "generate_ablation_report",
    "results_to_dataframe",
    "load_ablation_results",
    "format_ablation_summary",
]
