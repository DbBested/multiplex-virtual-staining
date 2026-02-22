"""
Validation report generation for biological metrics.

Generates LaTeX tables, co-expression heatmaps, and summary figures
for paper-ready biological validation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


def plot_coexpression_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Marker Co-expression",
    output_path: str = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot co-expression correlation matrix as heatmap.

    Args:
        corr_matrix: DataFrame with correlation values
        title: Plot title
        output_path: Path to save figure (optional)
        cmap: Colormap name
        vmin, vmax: Color scale limits
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson r"},
    )

    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_intensity_correlation(
    pred_intensities: pd.DataFrame,
    gt_intensities: pd.DataFrame,
    marker: str,
    output_path: str = None,
    figsize: tuple = (6, 6),
) -> plt.Figure:
    """Scatter plot of pred vs GT intensity for a marker.

    Args:
        pred_intensities: DataFrame with predicted intensities
        gt_intensities: DataFrame with GT intensities
        marker: Marker name to plot
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = gt_intensities[marker].values
    y = pred_intensities[marker].values

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    ax.scatter(x, y, alpha=0.5, s=10)

    # Fit line
    if len(x) > 2:
        from scipy.stats import pearsonr
        r, _ = pearsonr(x, y)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", label=f"r = {r:.3f}")
        ax.legend()

    ax.set_xlabel(f"Ground Truth {marker}")
    ax.set_ylabel(f"Predicted {marker}")
    ax.set_title(f"Per-cell Intensity: {marker}")

    # Identity line
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", alpha=0.3, label="y=x")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def generate_validation_report(
    results: dict,
    output_dir: str,
    include_figures: bool = True,
) -> dict:
    """Generate full biological validation report.

    Args:
        results: Output from PerCellMetrics.compute or compute_batch
        output_dir: Directory to save outputs
        include_figures: Generate and save figures

    Returns:
        dict with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1. Save JSON results
    json_path = output_dir / "biological_validation.json"

    # Convert DataFrames to dicts for JSON serialization
    json_results = _prepare_for_json(results)
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    outputs["json"] = str(json_path)

    # 2. Generate LaTeX table for intensity correlations
    latex_path = output_dir / "intensity_correlation.tex"
    latex_content = _generate_intensity_latex(results["intensity_correlation"])
    with open(latex_path, "w") as f:
        f.write(latex_content)
    outputs["latex_intensity"] = str(latex_path)

    # 3. Generate summary table
    summary_path = output_dir / "validation_summary.tex"
    summary_content = _generate_summary_latex(results)
    with open(summary_path, "w") as f:
        f.write(summary_content)
    outputs["latex_summary"] = str(summary_path)

    # 4. Generate figures
    if include_figures and "coexpression" in results:
        # Co-expression heatmaps
        coexp = results["coexpression"]
        if "pred_coexpression" in coexp:
            fig_pred = plot_coexpression_heatmap(
                coexp["pred_coexpression"],
                title="Predicted Marker Co-expression",
                output_path=output_dir / "coexp_pred.png",
            )
            plt.close(fig_pred)
            outputs["fig_coexp_pred"] = str(output_dir / "coexp_pred.png")

        if "real_coexpression" in coexp:
            fig_real = plot_coexpression_heatmap(
                coexp["real_coexpression"],
                title="Ground Truth Marker Co-expression",
                output_path=output_dir / "coexp_gt.png",
            )
            plt.close(fig_real)
            outputs["fig_coexp_gt"] = str(output_dir / "coexp_gt.png")

    return outputs


def _prepare_for_json(results: dict) -> dict:
    """Convert results to JSON-serializable format."""
    output = {}
    for key, value in results.items():
        # Convert tuple keys to strings
        str_key = str(key) if isinstance(key, tuple) else key
        if isinstance(value, pd.DataFrame):
            # Use 'split' orientation which handles multi-index better
            output[str_key] = value.reset_index(drop=True).to_dict(orient='list')
        elif isinstance(value, dict):
            output[str_key] = _prepare_for_json(value)
        elif isinstance(value, np.ndarray):
            output[str_key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            output[str_key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            output[str_key] = int(value)
        elif isinstance(value, np.bool_):
            output[str_key] = bool(value)
        elif isinstance(value, list):
            # Handle lists with numpy types
            output[str_key] = [
                float(v) if isinstance(v, (np.float32, np.float64)) else
                int(v) if isinstance(v, (np.int32, np.int64)) else
                bool(v) if isinstance(v, np.bool_) else v
                for v in value
            ]
        else:
            output[str_key] = value
    return output


def _generate_intensity_latex(intensity_corr: dict) -> str:
    """Generate LaTeX table for intensity correlations."""
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Per-cell intensity correlation between predicted and ground truth markers.}",
        "\\label{tab:intensity_correlation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Marker & Pearson $r$ & $n$ cells \\\\",
        "\\midrule",
    ]

    for marker, values in intensity_corr.items():
        if marker == "mean":
            continue
        r = values.get("pearson_r", np.nan)
        n = values.get("n_cells", 0)
        r_str = f"{r:.3f}" if not np.isnan(r) else "N/A"
        lines.append(f"{marker} & {r_str} & {n} \\\\")

    # Add mean row
    if "mean" in intensity_corr:
        mean_r = intensity_corr["mean"].get("pearson_r", np.nan)
        std_r = intensity_corr["mean"].get("pearson_r_std", np.nan)
        if not np.isnan(mean_r):
            lines.append("\\midrule")
            lines.append(f"\\textbf{{Mean}} & \\textbf{{{mean_r:.3f}}} $\\pm$ {std_r:.3f} & -- \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def _generate_summary_latex(results: dict) -> str:
    """Generate summary LaTeX table."""
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Biological validation summary.}",
        "\\label{tab:bio_validation_summary}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]

    # Number of cells
    n_cells = results.get("n_cells", 0)
    lines.append(f"Total cells analyzed & {n_cells} \\\\")

    # Dice score
    dice = results.get("segmentation", {}).get("mean_dice", results.get("mean_dice", np.nan))
    if not np.isnan(dice):
        lines.append(f"Segmentation Dice & {dice:.3f} \\\\")

    # Mean intensity correlation
    if "intensity_correlation" in results:
        mean_r = results["intensity_correlation"].get("mean", {}).get("pearson_r", np.nan)
        if not np.isnan(mean_r):
            lines.append(f"Mean intensity Pearson $r$ & {mean_r:.3f} \\\\")

    # Pattern correlation
    if "coexpression" in results:
        pattern_r = results["coexpression"].get("pattern_correlation", np.nan)
        if not np.isnan(pattern_r):
            lines.append(f"Co-expression pattern $r$ & {pattern_r:.3f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)
