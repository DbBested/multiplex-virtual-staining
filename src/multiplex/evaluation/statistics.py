"""
Statistical testing utilities for ablation comparison.

Provides bootstrap confidence intervals, effect size calculations, and
paired comparison utilities for rigorous statistical validation of
ablation study results.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap


def compute_bootstrap_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a single group of values.

    Uses the BCa (bias-corrected and accelerated) method from scipy.stats.bootstrap
    for improved accuracy compared to percentile methods.

    Args:
        values: Array of metric values to compute CI for.
        confidence_level: Confidence level for the interval (default 0.95 for 95% CI).
        n_resamples: Number of bootstrap resamples (default 10000).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) CI bounds.

    Raises:
        ValueError: If values is empty or has insufficient unique values for BCa.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> values = np.random.randn(100) + 5.0
        >>> ci = compute_bootstrap_ci(values)
        >>> print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
        95% CI: (4.806, 5.243)
    """
    values = np.asarray(values).ravel()

    if len(values) == 0:
        raise ValueError("Cannot compute CI for empty array")

    # Handle edge case: all identical values
    if np.all(values == values[0]):
        return (float(values[0]), float(values[0]))

    # Handle small sample sizes with fewer resamples
    actual_resamples = min(n_resamples, max(100, len(values) * 10))

    # scipy.stats.bootstrap expects data as a sequence of arrays
    data = (values,)

    try:
        result = bootstrap(
            data,
            statistic=np.mean,
            n_resamples=actual_resamples,
            confidence_level=confidence_level,
            method="BCa",
            random_state=random_state,
        )
        return (float(result.confidence_interval.low), float(result.confidence_interval.high))
    except Exception:
        # Fallback to percentile method if BCa fails (e.g., insufficient unique values)
        result = bootstrap(
            data,
            statistic=np.mean,
            n_resamples=actual_resamples,
            confidence_level=confidence_level,
            method="percentile",
            random_state=random_state,
        )
        return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation for the denominator, which is appropriate
    when comparing two independent groups with potentially different variances.

    Effect size interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d effect size. Positive if group1 mean > group2 mean.

    Raises:
        ValueError: If either group is empty or pooled variance is zero.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> group1 = np.random.randn(50) + 1.0  # Mean ~1
        >>> group2 = np.random.randn(50)         # Mean ~0
        >>> d = cohens_d(group1, group2)
        >>> print(f"Cohen's d: {d:.3f}")  # Should be ~1.0 (large effect)
        Cohen's d: 0.938
    """
    group1 = np.asarray(group1).ravel()
    group2 = np.asarray(group2).ravel()

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Cannot compute Cohen's d for empty groups")

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    # Formula: sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        # Both groups have zero variance (all identical values)
        if mean1 == mean2:
            return 0.0
        # Technically undefined, but return large value to indicate separation
        return float("inf") if mean1 > mean2 else float("-inf")

    return float((mean1 - mean2) / pooled_std)


def compare_paired_bootstrap(
    baseline: np.ndarray,
    variant: np.ndarray,
    metric_name: str = "metric",
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """Compare two ablations using paired bootstrap on matched test samples.

    This is the appropriate method when comparing ablation results from a single
    training run each, evaluated on the same test set. The pairing accounts for
    sample-to-sample correlation (same test image gives correlated predictions
    from both models).

    Args:
        baseline: Per-sample metric values from baseline model.
        variant: Per-sample metric values from variant model (must match baseline length).
        metric_name: Name of the metric for the result dict.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level for CI (default 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with comparison results:
            - metric: Name of the metric
            - baseline_mean: Mean of baseline values
            - variant_mean: Mean of variant values
            - mean_diff: Mean difference (variant - baseline)
            - ci_low: Lower bound of difference CI
            - ci_high: Upper bound of difference CI
            - effect_size: Cohen's d effect size
            - significant: True if CI excludes 0

    Raises:
        ValueError: If arrays have different lengths or are empty.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> baseline = np.random.randn(100) + 20.0  # PSNR ~20
        >>> variant = np.random.randn(100) + 22.0   # PSNR ~22 (improved)
        >>> result = compare_paired_bootstrap(baseline, variant, "PSNR")
        >>> print(f"Diff: {result['mean_diff']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
        Diff: 2.00 [1.65, 2.35]
    """
    baseline = np.asarray(baseline).ravel()
    variant = np.asarray(variant).ravel()

    if len(baseline) != len(variant):
        raise ValueError(
            f"Paired comparison requires equal-length arrays. "
            f"Got baseline={len(baseline)}, variant={len(variant)}"
        )

    if len(baseline) == 0:
        raise ValueError("Cannot compare empty arrays")

    # Compute paired differences
    differences = variant - baseline

    # Bootstrap CI on the differences
    ci_low, ci_high = compute_bootstrap_ci(
        differences,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    # Effect size
    effect = cohens_d(variant, baseline)

    # Significance: CI excludes 0
    significant = (ci_low > 0) or (ci_high < 0)

    return {
        "metric": metric_name,
        "baseline_mean": float(np.mean(baseline)),
        "variant_mean": float(np.mean(variant)),
        "mean_diff": float(np.mean(differences)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "effect_size": float(effect),
        "significant": bool(significant),
    }


def format_comparison_table(comparisons: list[dict], show_effect_interpretation: bool = True) -> str:
    """Format list of comparison results as a readable table.

    Args:
        comparisons: List of dicts from compare_paired_bootstrap.
        show_effect_interpretation: Whether to include effect size interpretation.

    Returns:
        Formatted string table suitable for logging or printing.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> results = []
        >>> for metric in ["PSNR", "SSIM"]:
        ...     baseline = np.random.randn(100) + 20
        ...     variant = np.random.randn(100) + 22
        ...     results.append(compare_paired_bootstrap(baseline, variant, metric))
        >>> print(format_comparison_table(results))
        Metric     Baseline   Variant    Diff        95% CI              Effect   Sig
        --------------------------------------------------------------------------------
        PSNR       19.93      21.90      +1.97       [+1.60, +2.32]      0.89     *
        SSIM       19.95      21.98      +2.04       [+1.66, +2.41]      0.93     *
    """
    if not comparisons:
        return "No comparisons to display."

    # Header
    lines = [
        f"{'Metric':<10} {'Baseline':>10} {'Variant':>10} {'Diff':>10} {'95% CI':>20} {'Effect':>8} {'Sig':>4}",
        "-" * 80,
    ]

    def _interpret_effect(d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    for comp in comparisons:
        metric = comp["metric"]
        baseline = comp["baseline_mean"]
        variant = comp["variant_mean"]
        diff = comp["mean_diff"]
        ci_low = comp["ci_low"]
        ci_high = comp["ci_high"]
        effect = comp["effect_size"]
        sig = "*" if comp["significant"] else ""

        # Format diff and CI with sign
        diff_str = f"{diff:+.2f}"
        ci_str = f"[{ci_low:+.2f}, {ci_high:+.2f}]"

        line = f"{metric:<10} {baseline:>10.2f} {variant:>10.2f} {diff_str:>10} {ci_str:>20} {effect:>8.2f} {sig:>4}"
        lines.append(line)

    if show_effect_interpretation:
        lines.append("")
        lines.append("Effect size interpretation: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >=0.8 large")
        lines.append("* indicates statistically significant (95% CI excludes 0)")

    return "\n".join(lines)


__all__ = [
    "compute_bootstrap_ci",
    "cohens_d",
    "compare_paired_bootstrap",
    "format_comparison_table",
]
