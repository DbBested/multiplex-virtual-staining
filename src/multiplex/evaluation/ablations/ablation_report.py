"""
Ablation report generation for LaTeX tables and figures.

This module provides functions to convert ablation results into
publication-ready LaTeX tables and JSON reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from multiplex.evaluation.latex_export import export_ablation_latex


def results_to_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert ablation results dictionary to pandas DataFrame.

    Args:
        results: Dictionary mapping ablation name to results dict.
            Each results dict should have 'metrics' -> {'mean' -> {metric: value}}.

    Returns:
        DataFrame with ablation names as index and metrics as columns.

    Example:
        >>> results = {
        ...     "full": {"metrics": {"mean": {"psnr": 28.5, "ssim": 0.87}}},
        ...     "no_attention": {"metrics": {"mean": {"psnr": 27.2, "ssim": 0.84}}},
        ... }
        >>> df = results_to_dataframe(results)
        >>> print(df)
                        psnr  ssim
        full           28.5  0.87
        no_attention   27.2  0.84
    """
    rows = []
    for name, result in results.items():
        if "error" in result:
            continue  # Skip failed experiments

        metrics = result.get("metrics", {})
        mean_metrics = metrics.get("mean", {})

        row = {"name": name}
        row.update(mean_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "name" in df.columns:
        df = df.set_index("name")

    return df


def load_ablation_results(results_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load ablation results from JSON files in directory.

    Supports both individual results.json files per ablation and
    a combined all_results.json file.

    Args:
        results_dir: Directory containing ablation results.

    Returns:
        Dictionary mapping ablation name to results dict.
    """
    results_dir = Path(results_dir)
    all_results = {}

    # Try loading combined results first
    combined_path = results_dir / "all_results.json"
    if combined_path.exists():
        with open(combined_path) as f:
            all_results = json.load(f)

    # Also load individual results (may override combined)
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            results_path = subdir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    result = json.load(f)
                    all_results[subdir.name] = result

    return all_results


def generate_ablation_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
    caption: str = "Ablation study results showing contribution of each component.",
    label: str = "tab:ablation",
) -> Dict[str, str]:
    """Generate ablation report with LaTeX table and JSON.

    Args:
        results: Dictionary mapping ablation name to results dict.
        output_dir: Directory for saving report files.
        caption: LaTeX table caption.
        label: LaTeX table label.

    Returns:
        Dictionary with paths to generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # Save full JSON results
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    output_paths["json"] = str(json_path)

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Save CSV
    csv_path = output_dir / "ablation_results.csv"
    df.to_csv(csv_path)
    output_paths["csv"] = str(csv_path)

    # Generate LaTeX table
    # Convert DataFrame to list of dicts for export_ablation_latex
    ablations = []
    for name in df.index:
        row = {"name": name}
        for col in df.columns:
            # Capitalize metric names for display
            display_name = col.upper() if col.lower() in ["psnr", "ssim", "lpips", "fid"] else col
            row[display_name] = df.loc[name, col]
        ablations.append(row)

    # Generate LaTeX
    latex_path = output_dir / "ablation_table.tex"
    latex_content = export_ablation_latex(
        ablations,
        str(latex_path),
        caption=caption,
        label=label,
    )
    output_paths["latex"] = str(latex_path)

    return output_paths


def format_ablation_summary(results: Dict[str, Dict[str, Any]]) -> str:
    """Format ablation results as a human-readable summary.

    Args:
        results: Dictionary mapping ablation name to results dict.

    Returns:
        Formatted string summary.
    """
    lines = ["Ablation Study Summary", "=" * 50, ""]

    df = results_to_dataframe(results)

    if df.empty:
        return "No ablation results available."

    # Find best for each metric
    best = {}
    for col in df.columns:
        if col.lower() in ["psnr", "ssim"]:
            best[col] = df[col].idxmax()
        elif col.lower() in ["lpips", "fid"]:
            best[col] = df[col].idxmin()

    # Format each ablation
    for name in df.index:
        result = results.get(name, {})
        description = result.get("description", "")
        train_time = result.get("train_time_seconds", 0)

        lines.append(f"{name}")
        lines.append(f"  Description: {description}")
        lines.append(f"  Training time: {train_time/3600:.1f}h")

        for col in df.columns:
            value = df.loc[name, col]
            is_best = best.get(col) == name
            marker = " *" if is_best else ""
            lines.append(f"  {col.upper()}: {value:.3f}{marker}")
        lines.append("")

    lines.append("* indicates best result for that metric")

    return "\n".join(lines)
