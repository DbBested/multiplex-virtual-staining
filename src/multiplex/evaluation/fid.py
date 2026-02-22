"""FID computation with feature caching for virtual staining evaluation.

This module provides FID computation using clean-fid with support for
caching real image statistics to enable efficient baseline comparison.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    fid = None
    CLEANFID_AVAILABLE = False

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    linalg = None
    SCIPY_AVAILABLE = False


def _stable_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet distance with numerical stability fix.

    Handles imaginary components that arise from numerical precision issues
    in the matrix square root computation.

    Args:
        mu1, sigma1: Mean and covariance of first distribution.
        mu2, sigma2: Mean and covariance of second distribution.
        eps: Small value added to diagonal for stability.

    Returns:
        Frechet distance (FID score).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular - add small epsilon to diagonal
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical instability - take real part if imaginary is small
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # Large imaginary component - add regularization and retry
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


MARKERS = ["LMNB1", "FBL", "TOMM20", "SEC61B", "TUBA1B"]


def _tensor_to_images(
    tensor: torch.Tensor,
    output_dir: str,
    prefix: str = "img",
) -> List[str]:
    """Save tensor to directory as PNG images.

    Args:
        tensor: Tensor of shape (N, C, H, W) in [0, 1] range.
        output_dir: Directory to save images.
        prefix: Filename prefix.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    tensor = tensor.detach().cpu()

    for i in range(tensor.shape[0]):
        for c in range(tensor.shape[1]):
            # Extract single channel, convert to uint8
            img_np = (tensor[i, c] * 255).numpy().astype(np.uint8)
            img = Image.fromarray(img_np, mode='L')

            # Convert to RGB (clean-fid expects RGB)
            img_rgb = img.convert('RGB')

            path = os.path.join(output_dir, f"{prefix}_{i:05d}_ch{c}.png")
            img_rgb.save(path)
            paths.append(path)

    return paths


def _save_marker_images(
    tensor: torch.Tensor,
    output_dir: str,
    marker_idx: int,
    prefix: str = "img",
) -> List[str]:
    """Save single marker channel as images.

    Args:
        tensor: Tensor of shape (N, C, H, W).
        output_dir: Output directory.
        marker_idx: Which marker channel to save.
        prefix: Filename prefix.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    tensor = tensor.detach().cpu()

    for i in range(tensor.shape[0]):
        img_np = (tensor[i, marker_idx] * 255).numpy().astype(np.uint8)
        img = Image.fromarray(img_np, mode='L')
        img_rgb = img.convert('RGB')

        path = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
        img_rgb.save(path)
        paths.append(path)

    return paths


class FIDComputer:
    """Compute FID with cached real image statistics.

    Caches Inception statistics for real images to enable efficient
    FID computation across multiple baselines.

    Args:
        real_images_dir: Directory containing real images, OR None
            to initialize later with cache_real_stats().
        cache_name: Name for cached statistics.
        cache_dir: Directory for cached stats (default: ~/.cache/cleanfid).
        mode: clean-fid mode ("clean" recommended).

    Example:
        >>> fid_computer = FIDComputer(
        ...     real_images_dir="./test_real/",
        ...     cache_name="multiplex_test"
        ... )
        >>> score = fid_computer.compute("./generated/")
    """

    def __init__(
        self,
        real_images_dir: Optional[str] = None,
        cache_name: str = "multiplex_test",
        cache_dir: Optional[str] = None,
        mode: str = "clean",
    ):
        if not CLEANFID_AVAILABLE:
            raise ImportError("clean-fid required: pip install clean-fid>=0.1.35")

        self.cache_name = cache_name
        self.cache_dir = cache_dir
        self.mode = mode
        self._stats_cached = False

        if real_images_dir is not None:
            self.cache_real_stats(real_images_dir)

    def cache_real_stats(self, real_images_dir: str) -> None:
        """Pre-compute and cache statistics for real images.

        Args:
            real_images_dir: Directory containing real PNG images.
        """
        if not os.path.isdir(real_images_dir):
            raise ValueError(f"Directory not found: {real_images_dir}")

        # Check if already cached
        if fid.test_stats_exists(self.cache_name, mode=self.mode):
            self._stats_cached = True
            return

        # Compute and cache stats
        fid.make_custom_stats(
            self.cache_name,
            real_images_dir,
            mode=self.mode,
            model_name="inception_v3",
        )
        self._stats_cached = True

    def compute(self, generated_dir: str) -> float:
        """Compute FID between generated images and cached real stats.

        Args:
            generated_dir: Directory containing generated PNG images.

        Returns:
            FID score (lower is better).
        """
        if not self._stats_cached:
            raise RuntimeError("Real stats not cached. Call cache_real_stats() first.")

        return fid.compute_fid(
            generated_dir,
            dataset_name=self.cache_name,
            mode=self.mode,
            dataset_split="custom",
        )

    def compute_between_dirs(
        self,
        real_dir: str,
        generated_dir: str,
    ) -> float:
        """Compute FID between two directories (no caching).

        Args:
            real_dir: Directory with real images.
            generated_dir: Directory with generated images.

        Returns:
            FID score.
        """
        return fid.compute_fid(real_dir, generated_dir, mode=self.mode)


def _compute_fid_stable(real_dir: str, gen_dir: str, mode: str = "clean") -> float:
    """Compute FID with numerical stability handling.

    Uses cleanfid to extract features but applies stable Frechet distance.

    Args:
        real_dir: Directory with real images.
        gen_dir: Directory with generated images.
        mode: cleanfid mode ("clean" recommended).

    Returns:
        FID score.
    """
    from cleanfid.features import build_feature_extractor, get_folder_features

    # Get feature extractor
    feat_model = build_feature_extractor(mode, "cuda" if torch.cuda.is_available() else "cpu")

    # Extract features
    np_feats_real = get_folder_features(real_dir, feat_model, num_workers=0, batch_size=32)
    np_feats_gen = get_folder_features(gen_dir, feat_model, num_workers=0, batch_size=32)

    # Compute statistics
    mu_real = np.mean(np_feats_real, axis=0)
    sigma_real = np.cov(np_feats_real, rowvar=False)
    mu_gen = np.mean(np_feats_gen, axis=0)
    sigma_gen = np.cov(np_feats_gen, rowvar=False)

    # Use stable Frechet distance
    return _stable_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_fid_from_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    markers: Optional[List[str]] = None,
    cleanup: bool = True,
) -> Dict[str, float]:
    """Compute FID from prediction and target tensors.

    Saves tensors to temporary directories and computes FID per marker.

    Args:
        pred: Predicted images (N, C, H, W) in [0, 1].
        target: Target images (N, C, H, W) in [0, 1].
        markers: List of marker names (defaults to MARKERS).
        cleanup: Whether to remove temp directories after computation.

    Returns:
        Dictionary with per-marker FID and mean FID.

    Example:
        >>> pred = torch.rand(100, 5, 256, 256)
        >>> target = torch.rand(100, 5, 256, 256)
        >>> results = compute_fid_from_tensors(pred, target)
        >>> print(results["mean"])  # Aggregated FID
    """
    if not CLEANFID_AVAILABLE:
        raise ImportError("clean-fid required: pip install clean-fid>=0.1.35")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for stable FID computation")

    if markers is None:
        markers = MARKERS

    num_markers = pred.shape[1]
    assert num_markers == len(markers), f"Expected {len(markers)} markers"

    results = {}
    temp_dirs = []

    try:
        for i, marker in enumerate(markers):
            # Create temp directories for this marker
            real_dir = tempfile.mkdtemp(prefix=f"fid_real_{marker}_")
            gen_dir = tempfile.mkdtemp(prefix=f"fid_gen_{marker}_")
            temp_dirs.extend([real_dir, gen_dir])

            # Save images
            _save_marker_images(target, real_dir, i, prefix="real")
            _save_marker_images(pred, gen_dir, i, prefix="gen")

            # Compute FID for this marker with numerical stability fix
            fid_score = _compute_fid_stable(real_dir, gen_dir, mode="clean")
            results[marker] = fid_score

        # Compute mean FID across markers
        results["mean"] = float(np.mean([results[m] for m in markers]))
        results["std"] = float(np.std([results[m] for m in markers]))

    finally:
        if cleanup:
            for d in temp_dirs:
                if os.path.exists(d):
                    shutil.rmtree(d)

    return results


def compute_fid_per_marker(
    real_tensors: List[torch.Tensor],
    gen_tensors: List[torch.Tensor],
    markers: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute FID from lists of tensors (accumulating batches).

    Use this when processing data in batches but need FID over full set.

    Args:
        real_tensors: List of real image tensors (each N, C, H, W).
        gen_tensors: List of generated image tensors.
        markers: Marker names.

    Returns:
        Per-marker and mean FID scores.
    """
    # Concatenate all tensors
    real = torch.cat(real_tensors, dim=0)
    gen = torch.cat(gen_tensors, dim=0)

    return compute_fid_from_tensors(real, gen, markers=markers)
