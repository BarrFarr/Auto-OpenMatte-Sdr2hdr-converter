"""Spatial/seam metrics: gradient continuity, edge preservation, seam error.

Measures quality at the HDR/extension boundary (seam) in open matte composites.
All measurements are performed in a band straddling the seam boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import compute_luminance, BT2020_LUMA


@dataclass
class SeamMetrics:
    """Combined seam quality metrics."""

    luminance_gradient_diff: float
    chroma_gradient_diff: float
    continuity_score: float  # 0-1, higher is better


def gradient_continuity(
    composite: NDArray[np.floating],
    seam_row: int,
    band_width: int = 8,
) -> float:
    """Measure luminance gradient continuity at the seam boundary.

    Computes the average absolute difference in vertical luminance gradient
    across the seam, within a band of specified width.

    Args:
        composite: Composite image in linear BT.2020 RGB, shape (H, W, 3).
        seam_row: Row index of the seam boundary.
        band_width: Width of the measurement band on each side of the seam.

    Returns:
        Average absolute gradient difference at the seam (lower is better).
    """
    lum = compute_luminance(composite, BT2020_LUMA)

    # Ensure we have enough rows
    top_start = max(0, seam_row - band_width)
    bot_end = min(lum.shape[0], seam_row + band_width)

    if top_start >= seam_row or seam_row >= bot_end:
        return 0.0

    # Vertical gradient (difference between adjacent rows)
    grad = np.diff(lum, axis=0)

    # Gradient just above seam vs just below seam
    grad_above = grad[seam_row - 1] if seam_row > 0 else np.zeros(lum.shape[1])
    grad_below = grad[seam_row] if seam_row < grad.shape[0] else np.zeros(lum.shape[1])

    # Average absolute gradient mismatch
    diff = np.abs(grad_above - grad_below)
    return float(np.mean(diff))


def edge_preservation(
    composite: NDArray[np.floating],
    reference: NDArray[np.floating],
    seam_row: int,
    band_width: int = 8,
) -> float:
    """Measure edge structure preservation in the seam band.

    Compares horizontal gradient magnitudes in the seam band between composite
    and reference to assess if edges are preserved during reshaping.

    Args:
        composite: Composite image, shape (H, W, 3).
        reference: Reference (ground truth) image, same shape.
        seam_row: Row index of the seam boundary.
        band_width: Width of the measurement band on each side.

    Returns:
        Edge preservation ratio (1.0 = perfect, < 1.0 = edges lost).
    """
    lum_comp = compute_luminance(composite, BT2020_LUMA)
    lum_ref = compute_luminance(reference, BT2020_LUMA)

    top = max(0, seam_row - band_width)
    bot = min(lum_comp.shape[0], seam_row + band_width)

    if top >= bot:
        return 1.0

    # Horizontal gradients in the band
    band_comp = lum_comp[top:bot, :]
    band_ref = lum_ref[top:bot, :]

    grad_comp = np.abs(np.diff(band_comp, axis=1))
    grad_ref = np.abs(np.diff(band_ref, axis=1))

    # Edge preservation: ratio of gradient magnitudes
    ref_energy = np.mean(grad_ref)
    if ref_energy < 1e-10:
        return 1.0

    comp_energy = np.mean(grad_comp)
    # Ratio clamped to [0, 2] to avoid extreme outliers
    ratio = min(comp_energy / ref_energy, 2.0)
    # Convert to preservation score (1.0 = perfect match)
    return float(1.0 - abs(1.0 - ratio))


def seam_error(
    composite: NDArray[np.floating],
    seam_row: int,
    band_width: int = 8,
) -> SeamMetrics:
    """Comprehensive seam quality measurement.

    Computes luminance gradient difference, chroma gradient difference,
    and an overall continuity score in the seam band.

    Args:
        composite: Composite image in linear BT.2020 RGB, shape (H, W, 3).
        seam_row: Row index of the seam boundary.
        band_width: Width of measurement band on each side.

    Returns:
        SeamMetrics dataclass with all seam quality measurements.
    """
    H, W, _ = composite.shape
    lum = compute_luminance(composite, BT2020_LUMA)

    # Compute chroma as sqrt(sum of squared color differences from luminance)
    lum_expanded = lum[..., np.newaxis]
    chroma = np.sqrt(np.mean((composite - lum_expanded) ** 2, axis=-1))

    # Luminance gradient difference at seam
    lum_grad = np.diff(lum, axis=0)
    if seam_row > 0 and seam_row < H - 1:
        lum_grad_above = lum_grad[seam_row - 1]
        lum_grad_below = lum_grad[seam_row]
        lum_grad_diff = float(np.mean(np.abs(lum_grad_above - lum_grad_below)))
    else:
        lum_grad_diff = 0.0

    # Chroma gradient difference at seam
    chroma_grad = np.diff(chroma, axis=0)
    if seam_row > 0 and seam_row < H - 1:
        chroma_grad_above = chroma_grad[seam_row - 1]
        chroma_grad_below = chroma_grad[seam_row]
        chroma_grad_diff = float(np.mean(np.abs(chroma_grad_above - chroma_grad_below)))
    else:
        chroma_grad_diff = 0.0

    # Overall continuity score: based on how smooth the seam band is
    top = max(0, seam_row - band_width)
    bot = min(H, seam_row + band_width)

    if top >= bot or bot - top < 3:
        continuity = 1.0
    else:
        band_lum = lum[top:bot, :]
        # Compute variance of vertical gradient within the band
        band_grad = np.diff(band_lum, axis=0)
        grad_variance = np.var(band_grad)
        # Score: exp(-variance) gives 1.0 for perfectly smooth, approaches 0 for rough
        continuity = float(np.exp(-grad_variance * 100.0))

    return SeamMetrics(
        luminance_gradient_diff=lum_grad_diff,
        chroma_gradient_diff=chroma_grad_diff,
        continuity_score=continuity,
    )
