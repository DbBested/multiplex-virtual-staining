"""Visual comparison figure generation for virtual staining evaluation.

This module provides tools to create publication-ready comparison grids
showing input, predicted, ground truth, and error maps.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import ImageGrid

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    sns = None
    SNS_AVAILABLE = False


MARKERS = ["LMNB1", "FBL", "TOMM20", "SEC61B", "TUBA1B"]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _normalize_for_display(
    img: np.ndarray,
    percentile_clip: Tuple[float, float] = (1, 99),
) -> np.ndarray:
    """Normalize image for display with percentile clipping.

    Args:
        img: Image array.
        percentile_clip: Low and high percentiles for clipping.

    Returns:
        Normalized image in [0, 1] range.
    """
    low, high = np.percentile(img, percentile_clip)
    if high - low < 1e-8:
        return np.zeros_like(img)
    img_clipped = np.clip(img, low, high)
    return (img_clipped - low) / (high - low)


def create_comparison_grid(
    samples: List[Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    dpi: int = 300,
    figsize_per_sample: Tuple[float, float] = (3.0, 3.0),
    marker_idx: int = 0,
    marker_name: Optional[str] = None,
    show_colorbar: bool = True,
) -> Figure:
    """Create comparison grid for multiple samples.

    Each sample shows: Input BF, Predicted, Ground Truth, Error Map.

    Args:
        samples: List of dicts with keys 'bf', 'pred', 'target'.
        save_path: Path to save figure (optional).
        dpi: Resolution for saved figure.
        figsize_per_sample: Size per sample row.
        marker_idx: Which marker channel to visualize.
        marker_name: Name for title (defaults to MARKERS[marker_idx]).
        show_colorbar: Whether to show colorbar on error map.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> samples = [{'bf': bf_np, 'pred': pred_np, 'target': gt_np}, ...]
        >>> fig = create_comparison_grid(samples, 'comparison.png')
    """
    n_samples = len(samples)
    n_cols = 4  # Input, Pred, GT, Error

    if marker_name is None:
        marker_name = MARKERS[marker_idx] if marker_idx < len(MARKERS) else f"Ch{marker_idx}"

    fig = plt.figure(figsize=(figsize_per_sample[0] * n_cols, figsize_per_sample[1] * n_samples))

    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(n_samples, n_cols),
        axes_pad=0.1,
        share_all=True,
        cbar_location="right" if show_colorbar else None,
        cbar_mode="single" if show_colorbar else None,
        cbar_size="5%",
        cbar_pad=0.1,
    )

    col_titles = ["Input (BF)", f"Predicted ({marker_name})", f"GT ({marker_name})", "Error Map"]

    for i, sample in enumerate(samples):
        bf = _to_numpy(sample['bf'])
        pred = _to_numpy(sample['pred'])
        target = _to_numpy(sample['target'])

        # Extract single channel if multi-channel
        if bf.ndim == 3:
            bf = bf[0]  # First channel
        if pred.ndim == 3:
            pred = pred[marker_idx]
        if target.ndim == 3:
            target = target[marker_idx]

        # Normalize for display
        bf_display = _normalize_for_display(bf)
        pred_display = _normalize_for_display(pred)
        target_display = _normalize_for_display(target)

        # Error map (absolute difference)
        error = np.abs(pred_display - target_display)

        images = [bf_display, pred_display, target_display, error]
        cmaps = ['gray', 'gray', 'gray', 'hot']

        for j, (img, cmap) in enumerate(zip(images, cmaps)):
            ax = grid[i * n_cols + j]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if j < 3 else None)
            ax.axis('off')

            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, fontweight='bold')

            if j == 3 and show_colorbar and i == n_samples - 1:
                grid.cbar_axes[0].colorbar(im)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

    return fig


def create_multi_marker_grid(
    sample: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    dpi: int = 300,
    markers: Optional[List[str]] = None,
) -> Figure:
    """Create grid showing all markers for a single sample.

    Layout: rows = markers, cols = [Predicted, GT, Error].

    Args:
        sample: Dict with 'bf', 'pred', 'target'.
        save_path: Path to save figure.
        dpi: Resolution.
        markers: Marker names.

    Returns:
        Figure object.
    """
    if markers is None:
        markers = MARKERS

    pred = _to_numpy(sample['pred'])
    target = _to_numpy(sample['target'])
    bf = _to_numpy(sample['bf'])

    n_markers = pred.shape[0] if pred.ndim == 3 else 1
    n_cols = 4  # BF, Pred, GT, Error

    fig, axes = plt.subplots(
        n_markers, n_cols,
        figsize=(3 * n_cols, 3 * n_markers),
        squeeze=False,
    )

    # First column is BF (same for all rows)
    if bf.ndim == 3:
        bf = bf[0]
    bf_display = _normalize_for_display(bf)

    for row, marker in enumerate(markers[:n_markers]):
        p = pred[row] if pred.ndim == 3 else pred
        t = target[row] if target.ndim == 3 else target

        p_display = _normalize_for_display(p)
        t_display = _normalize_for_display(t)
        error = np.abs(p_display - t_display)

        # BF column
        axes[row, 0].imshow(bf_display, cmap='gray')
        axes[row, 0].set_ylabel(marker, fontsize=10, fontweight='bold')

        # Pred column
        axes[row, 1].imshow(p_display, cmap='gray')

        # GT column
        axes[row, 2].imshow(t_display, cmap='gray')

        # Error column
        im = axes[row, 3].imshow(error, cmap='hot')

        for ax in axes[row]:
            ax.axis('off')

    # Column titles
    axes[0, 0].set_title("Input (BF)", fontsize=10)
    axes[0, 1].set_title("Predicted", fontsize=10)
    axes[0, 2].set_title("Ground Truth", fontsize=10)
    axes[0, 3].set_title("Error", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

    return fig


class ComparisonVisualizer:
    """High-level interface for generating comparison figures.

    Args:
        output_dir: Directory for saving figures.
        dpi: Resolution for saved figures.
        format: Image format ('png', 'pdf', 'svg').

    Example:
        >>> viz = ComparisonVisualizer('./figures/', dpi=300)
        >>> viz.save_comparison_grid(samples, 'fig1_comparison.png')
    """

    def __init__(
        self,
        output_dir: str = "./figures",
        dpi: int = 300,
        format: str = "png",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.format = format

    def save_comparison_grid(
        self,
        samples: List[Dict[str, np.ndarray]],
        filename: str,
        **kwargs,
    ) -> str:
        """Save comparison grid to file.

        Args:
            samples: List of sample dicts.
            filename: Output filename.
            **kwargs: Passed to create_comparison_grid.

        Returns:
            Path to saved file.
        """
        save_path = self.output_dir / filename
        create_comparison_grid(
            samples,
            save_path=str(save_path),
            dpi=self.dpi,
            **kwargs,
        )
        return str(save_path)

    def save_multi_marker_grid(
        self,
        sample: Dict[str, np.ndarray],
        filename: str,
        **kwargs,
    ) -> str:
        """Save multi-marker grid for single sample."""
        save_path = self.output_dir / filename
        create_multi_marker_grid(
            sample,
            save_path=str(save_path),
            dpi=self.dpi,
            **kwargs,
        )
        return str(save_path)

    def save_metrics_barplot(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str,
        metric: str = "ssim",
    ) -> str:
        """Create bar plot comparing methods on a metric.

        Args:
            results: Dict of method -> {metric: value}.
            filename: Output filename.
            metric: Which metric to plot.

        Returns:
            Path to saved file.
        """
        if SNS_AVAILABLE:
            sns.set_style("whitegrid")

        methods = list(results.keys())
        values = [results[m].get(metric, 0) for m in methods]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(methods, values, color='steelblue', edgecolor='black')

        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"{metric.upper()} Comparison", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                fontsize=9,
            )

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def save_constraint_violations(
        self,
        pred: np.ndarray,
        filename: str,
        **kwargs,
    ) -> str:
        """Save constraint violation visualization.

        Args:
            pred: (5, H, W) predicted markers
            filename: Output filename
            **kwargs: Passed to visualize_constraint_violations

        Returns:
            Path to saved file
        """
        save_path = self.output_dir / filename
        visualize_constraint_violations(
            pred,
            save_path=str(save_path),
            dpi=self.dpi,
            **kwargs,
        )
        return str(save_path)


def visualize_constraint_violations(
    pred: np.ndarray,
    save_path: Optional[str] = None,
    markers: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300,
) -> Figure:
    """Visualize biological constraint violations in predictions.

    Creates a 2x2 grid showing:
    - Top-left: FBL containment (nuclear region with FBL overlay, red=outside)
    - Top-right: TOMM20 exclusion (cytoplasmic marker with nuclear region, red=violation)
    - Bottom-left: Mito-ER colocalization (TOMM20 red / SEC61B green composite)
    - Bottom-right: Summary metrics text

    Args:
        pred: (5, H, W) predicted marker images in [0, 1] range
        save_path: Optional path to save figure
        markers: Marker names (default: MARKERS)
        figsize: Figure size
        dpi: Resolution for saved figure

    Returns:
        Matplotlib Figure object
    """
    from scipy.ndimage import binary_fill_holes

    if markers is None:
        markers = MARKERS

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Constants for marker indices
    LMNB1_IDX, FBL_IDX, TOMM20_IDX, SEC61B_IDX = 0, 1, 2, 3

    # Create nuclear mask
    nuclear_mask = pred[LMNB1_IDX] > 0.5
    nuclear_interior = binary_fill_holes(nuclear_mask)

    # 1. FBL Containment (top-left)
    ax = axes[0, 0]
    ax.imshow(pred[LMNB1_IDX], cmap='gray', vmin=0, vmax=1)
    # Overlay FBL in cyan
    fbl_mask = pred[FBL_IDX] > 0.3
    fbl_overlay = np.ma.masked_where(~fbl_mask, pred[FBL_IDX])
    ax.imshow(fbl_overlay, cmap='cool', alpha=0.7, vmin=0, vmax=1)
    # Highlight FBL outside nuclear region in red
    fbl_outside = fbl_mask & ~nuclear_interior
    if fbl_outside.any():
        ax.contour(fbl_outside, colors='red', levels=[0.5], linewidths=2)
    ax.set_title("FBL Containment\n(cyan=FBL, red=outside nucleus)", fontsize=10)
    ax.axis('off')

    # 2. TOMM20 Exclusion (top-right)
    ax = axes[0, 1]
    ax.imshow(pred[TOMM20_IDX], cmap='gray', vmin=0, vmax=1)
    # Show nuclear boundary in cyan
    ax.contour(nuclear_interior, colors='cyan', levels=[0.5], linewidths=1.5)
    # Highlight TOMM20 inside nucleus in red
    tomm20_mask = pred[TOMM20_IDX] > 0.3
    tomm20_inside = tomm20_mask & nuclear_interior
    if tomm20_inside.any():
        ax.contour(tomm20_inside, colors='red', levels=[0.5], linewidths=2)
    ax.set_title("TOMM20 Exclusion\n(cyan=nucleus, red=violation)", fontsize=10)
    ax.axis('off')

    # 3. Mito-ER Colocalization (bottom-left)
    ax = axes[1, 0]
    composite = np.zeros((*pred[0].shape, 3))
    # Normalize each channel for display
    tomm20_norm = _normalize_for_display(pred[TOMM20_IDX])
    sec61b_norm = _normalize_for_display(pred[SEC61B_IDX])
    composite[:, :, 0] = tomm20_norm  # TOMM20 in red
    composite[:, :, 1] = sec61b_norm  # SEC61B in green
    # Yellow = overlap (mito-ER contacts)
    ax.imshow(composite)
    ax.set_title("Mito-ER Colocalization\n(R=TOMM20, G=SEC61B, Y=overlap)", fontsize=10)
    ax.axis('off')

    # 4. Summary (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')

    # Compute quick metrics for display
    # Exclusion: fraction of TOMM20 outside nucleus
    tomm20_total = pred[TOMM20_IDX].sum()
    tomm20_outside = (pred[TOMM20_IDX] * ~nuclear_interior).sum()
    exclusion_pct = (tomm20_outside / (tomm20_total + 1e-8)) * 100

    # Containment: fraction of FBL inside nucleus
    fbl_total = pred[FBL_IDX].sum()
    fbl_inside = (pred[FBL_IDX] * nuclear_interior).sum()
    containment_pct = (fbl_inside / (fbl_total + 1e-8)) * 100

    # Colocalization: Pearson r
    tomm20_flat = pred[TOMM20_IDX].flatten()
    sec61b_flat = pred[SEC61B_IDX].flatten()
    tomm20_c = tomm20_flat - tomm20_flat.mean()
    sec61b_c = sec61b_flat - sec61b_flat.mean()
    r = (tomm20_c * sec61b_c).sum() / (
        np.sqrt((tomm20_c**2).sum()) * np.sqrt((sec61b_c**2).sum()) + 1e-8
    )

    summary_text = (
        "Constraint Summary\n"
        "==================\n\n"
        f"TOMM20 outside nucleus: {exclusion_pct:.1f}%\n"
        f"  (higher = better exclusion)\n\n"
        f"FBL inside nucleus: {containment_pct:.1f}%\n"
        f"  (higher = better containment)\n\n"
        f"TOMM20-SEC61B correlation: r={r:.3f}\n"
        f"  (positive = mito-ER proximity)"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()

    return fig
