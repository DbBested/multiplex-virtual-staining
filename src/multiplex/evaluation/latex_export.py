"""LaTeX table export for evaluation results.

This module provides functions to export evaluation results as
publication-ready LaTeX tables with booktabs formatting.

Note: Generates LaTeX manually without jinja2 dependency.
"""

from typing import Any, Dict, List, Optional, Union


def format_metric_cell(
    value: float,
    std: Optional[float] = None,
    bold: bool = False,
    precision: int = 3,
) -> str:
    """Format a metric value for LaTeX table.

    Args:
        value: Metric value.
        std: Standard deviation (optional).
        bold: Whether to bold the value (for best results).
        precision: Decimal precision.

    Returns:
        Formatted string for LaTeX.

    Example:
        >>> format_metric_cell(0.856, std=0.02, bold=True)
        '\\textbf{0.856} $\\pm$ 0.020'
    """
    formatted = f"{value:.{precision}f}"
    if bold:
        formatted = f"\\textbf{{{formatted}}}"
    if std is not None:
        formatted = f"{formatted} $\\pm$ {std:.{precision}f}"
    return formatted


def _find_best_values(
    results: Dict[str, Dict[str, float]],
    higher_is_better: Dict[str, bool],
) -> Dict[str, str]:
    """Find best method for each metric.

    Args:
        results: Dict of method -> {metric: value}.
        higher_is_better: Dict mapping metric name to bool.

    Returns:
        Dict mapping metric name to best method name.
    """
    best = {}
    # Get all metrics from first result
    if not results:
        return best

    first_method = next(iter(results.keys()))
    metrics = list(results[first_method].keys())

    for metric in metrics:
        values = {method: results[method].get(metric, 0) for method in results}
        is_higher_better = higher_is_better.get(metric, True)

        if is_higher_better:
            best[metric] = max(values, key=values.get)
        else:
            best[metric] = min(values, key=values.get)

    return best


def _generate_tabular(
    methods: List[str],
    metrics: List[str],
    formatted_values: Dict[str, Dict[str, str]],
) -> str:
    """Generate LaTeX tabular content with booktabs.

    Args:
        methods: List of method names.
        metrics: List of metric names.
        formatted_values: method -> metric -> formatted string.

    Returns:
        LaTeX tabular string.
    """
    n_cols = len(metrics)
    col_format = "l" + "c" * n_cols

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")

    # Header row
    header = " & ".join(["Method"] + metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for method in methods:
        row_values = [formatted_values[method].get(m, "-") for m in metrics]
        row = method + " & " + " & ".join(row_values) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def export_results_latex(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    caption: str = "Quantitative comparison of image translation methods.",
    label: str = "tab:results",
    higher_is_better: Optional[Dict[str, bool]] = None,
    include_std: bool = False,
    std_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Export results dictionary to LaTeX table.

    Args:
        results: Dict of method -> {metric: value}.
        output_path: Path to save .tex file.
        caption: Table caption.
        label: LaTeX label for referencing.
        higher_is_better: Dict indicating if higher is better per metric.
            Default: PSNR=True, SSIM=True, LPIPS=False, FID=False.
        include_std: Whether to include standard deviation.
        std_results: Dict of method -> {metric: std_value} if include_std.

    Returns:
        LaTeX table as string.

    Example:
        >>> results = {
        ...     "Ours": {"PSNR": 28.5, "SSIM": 0.87, "LPIPS": 0.12},
        ...     "pix2pix": {"PSNR": 25.3, "SSIM": 0.82, "LPIPS": 0.18},
        ... }
        >>> latex = export_results_latex(results, "table.tex")
    """
    # Default higher_is_better
    if higher_is_better is None:
        higher_is_better = {
            "PSNR": True,
            "psnr": True,
            "SSIM": True,
            "ssim": True,
            "LPIPS": False,
            "lpips": False,
            "FID": False,
            "fid": False,
        }

    # Find best per column
    best = _find_best_values(results, higher_is_better)

    # Get ordered methods and metrics
    methods = list(results.keys())
    if methods:
        metrics = list(results[methods[0]].keys())
    else:
        metrics = []

    # Format cells with bolding for best
    formatted_values: Dict[str, Dict[str, str]] = {}
    for method in methods:
        formatted_values[method] = {}
        for metric in metrics:
            value = results[method].get(metric, 0)
            is_best = (metric in best and best[metric] == method)
            std = None
            if include_std and std_results and method in std_results:
                std = std_results[method].get(metric)
            formatted_values[method][metric] = format_metric_cell(
                value, std=std, bold=is_best
            )

    # Generate tabular
    tabular = _generate_tabular(methods, metrics, formatted_values)

    # Wrap in table environment
    full_latex = f"""% Requires \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{tabular}
\\end{{table}}
"""

    # Save to file
    with open(output_path, 'w') as f:
        f.write(full_latex)

    return full_latex


def export_per_marker_latex(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    metric: str = "SSIM",
    caption: Optional[str] = None,
    label: str = "tab:per_marker",
) -> str:
    """Export per-marker results for a single metric.

    Args:
        results: Dict of method -> marker -> {metric: value}.
        output_path: Output path.
        metric: Which metric to export.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if caption is None:
        caption = f"Per-marker {metric} comparison."

    # Restructure: method -> {marker: value}
    data: Dict[str, Dict[str, float]] = {}
    markers_set = set()
    for method, markers_dict in results.items():
        data[method] = {}
        for mk, vals in markers_dict.items():
            if mk not in ['mean', 'std']:
                data[method][mk] = vals.get(metric, 0)
                markers_set.add(mk)
        data[method]['Mean'] = markers_dict.get('mean', {}).get(metric, 0)

    markers = sorted(list(markers_set)) + ['Mean']

    # Find best per marker
    higher = metric.upper() in ['PSNR', 'SSIM']
    best = {}
    for mk in markers:
        values = {method: data[method].get(mk, 0) for method in data}
        best[mk] = max(values, key=values.get) if higher else min(values, key=values.get)

    # Format with bolding
    methods = list(data.keys())
    formatted_values: Dict[str, Dict[str, str]] = {}
    for method in methods:
        formatted_values[method] = {}
        for mk in markers:
            val = data[method].get(mk, 0)
            is_best = (best.get(mk) == method)
            formatted_values[method][mk] = format_metric_cell(val, bold=is_best)

    # Generate tabular
    tabular = _generate_tabular(methods, markers, formatted_values)

    full_latex = f"""% Requires \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{tabular}
\\end{{table}}
"""

    with open(output_path, 'w') as f:
        f.write(full_latex)

    return full_latex


def export_ablation_latex(
    ablations: List[Dict[str, Any]],
    output_path: str,
    caption: str = "Ablation study results.",
    label: str = "tab:ablation",
) -> str:
    """Export ablation study results.

    Args:
        ablations: List of dicts with 'name', 'config', and metrics.
        output_path: Output path.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX string.

    Example:
        >>> ablations = [
        ...     {"name": "Full model", "PSNR": 28.5, "SSIM": 0.87},
        ...     {"name": "w/o attention", "PSNR": 27.2, "SSIM": 0.84},
        ... ]
    """
    # Extract metric columns (exclude 'name' and 'config')
    if not ablations:
        return ""

    metric_cols = [k for k in ablations[0].keys() if k not in ['name', 'config']]
    names = [a['name'] for a in ablations]

    # Convert to results format
    results: Dict[str, Dict[str, float]] = {}
    for ablation in ablations:
        name = ablation['name']
        results[name] = {col: ablation.get(col, 0) for col in metric_cols}

    # Find best for each metric (higher is better except LPIPS/FID)
    higher_is_better = {col: col.lower() not in ['lpips', 'fid'] for col in metric_cols}
    best = _find_best_values(results, higher_is_better)

    # Format with bolding
    formatted_values: Dict[str, Dict[str, str]] = {}
    for name in names:
        formatted_values[name] = {}
        for col in metric_cols:
            val = results[name].get(col, 0)
            is_best = (best.get(col) == name)
            formatted_values[name][col] = format_metric_cell(val, bold=is_best)

    # Generate tabular
    tabular = _generate_tabular(names, metric_cols, formatted_values)

    full_latex = f"""% Requires \\usepackage{{booktabs}}
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{tabular}
\\end{{table}}
"""

    with open(output_path, 'w') as f:
        f.write(full_latex)

    return full_latex
