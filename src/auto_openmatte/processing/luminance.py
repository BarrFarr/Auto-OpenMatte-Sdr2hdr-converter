"""Luminance mapping estimation.

Derives the SDR → HDR luminance transfer function from the overlap region.
Produces a monotonic mapping that can be applied to the extension areas.

Key principles:
- HDR is MASTER — we map FROM SDR TO HDR, never the reverse
- Mapping must be monotonic (no inversions)
- Uses robust statistics (percentile bins, outlier rejection)
- One stable transform per shot (not per frame)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import Shot, ShotTransform
from auto_openmatte.utils.math_utils import (
    fit_monotonic_spline,
    percentile_bins,
)

logger = logging.getLogger(__name__)


def estimate_luminance_curve(
    sdr_luminance: NDArray[np.floating],
    hdr_luminance: NDArray[np.floating],
    config: ColorConfig | None = None,
) -> list[list[float]]:
    """Estimate the SDR → HDR luminance mapping curve.

    Uses percentile binning with robust median estimation and monotonic
    enforcement to produce a stable, invertible mapping.

    Args:
        sdr_luminance: Flattened SDR luminance values (linear, [0, 1]).
        hdr_luminance: Corresponding HDR luminance values (linear, normalized).
        config: Color configuration.

    Returns:
        List of [sdr_value, hdr_value] control points for the mapping curve.
        Guaranteed to be monotonically non-decreasing.
    """
    if config is None:
        config = ColorConfig()

    # Step 1: Remove obvious outliers (clipped values, noise)
    valid_mask = (
        (sdr_luminance >= 0.0)
        & (sdr_luminance <= 1.0)
        & (hdr_luminance >= 0.0)
        & np.isfinite(sdr_luminance)
        & np.isfinite(hdr_luminance)
    )

    sdr_clean = sdr_luminance[valid_mask]
    hdr_clean = hdr_luminance[valid_mask]

    if len(sdr_clean) < 100:
        logger.warning("Too few valid samples for luminance estimation")
        # Return identity-like curve
        return [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

    # Step 2: Percentile clipping (exclude extreme shadows and highlights)
    percentile_range = (config.low_percentile, config.high_percentile)

    # Step 3: Bin and compute robust medians
    bin_centers, bin_medians = percentile_bins(
        sdr_clean, hdr_clean,
        n_bins=config.luminance_bins,
        percentile_range=percentile_range,
    )

    if len(bin_centers) < 5:
        logger.warning("Too few valid bins for luminance curve")
        return [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

    # Step 4: Enforce monotonicity
    x_mono, y_mono = fit_monotonic_spline(bin_centers, bin_medians)

    # Step 5: Subsample to reasonable number of control points
    # Keep 32-64 points for the spline (more than enough for smooth interpolation)
    n_points = min(64, len(x_mono))
    indices = np.linspace(0, len(x_mono) - 1, n_points, dtype=int)
    x_final = x_mono[indices]
    y_final = y_mono[indices]

    # Build control point list
    curve = [[float(x), float(y)] for x, y in zip(x_final, y_final)]

    # Ensure curve starts near 0 and end point is reasonable
    if curve[0][0] > 0.01:
        curve.insert(0, [0.0, 0.0])
    if curve[-1][0] < 0.99:
        # Extrapolate last point
        curve.append([1.0, curve[-1][1] * 1.05])  # Slight extrapolation

    logger.info(
        f"Luminance curve: {len(curve)} control points, "
        f"range [{curve[0][0]:.3f}, {curve[-1][0]:.3f}] -> "
        f"[{curve[0][1]:.3f}, {curve[-1][1]:.3f}]"
    )
    return curve


def apply_luminance_curve(
    sdr_luminance: NDArray[np.floating],
    curve: list[list[float]],
) -> NDArray[np.floating]:
    """Apply a luminance mapping curve to SDR values.

    Uses PCHIP (monotonic cubic) interpolation between control points.

    Args:
        sdr_luminance: Input SDR luminance values (linear, [0, 1]).
        curve: Control points [[sdr_val, hdr_val], ...].

    Returns:
        Mapped HDR luminance values.
    """
    if not curve or len(curve) < 2:
        return sdr_luminance.copy()

    from scipy.interpolate import PchipInterpolator

    x_points = np.array([p[0] for p in curve])
    y_points = np.array([p[1] for p in curve])

    # PCHIP guarantees monotonicity between control points
    interpolator = PchipInterpolator(x_points, y_points, extrapolate=True)

    result = interpolator(sdr_luminance)
    # Clamp to non-negative
    return np.maximum(result, 0.0)


def estimate_shot_luminance(
    sdr_samples: list[NDArray[np.floating]],
    hdr_samples: list[NDArray[np.floating]],
    shot: Shot,
    config: ColorConfig | None = None,
) -> ShotTransform:
    """Estimate luminance transform for a single shot.

    Combines samples from multiple frames within the shot for stability.

    Args:
        sdr_samples: List of SDR luminance arrays (one per sample frame).
        hdr_samples: List of corresponding HDR luminance arrays.
        shot: The shot being analyzed.
        config: Color configuration.

    Returns:
        ShotTransform with luminance_curve populated.
    """
    if config is None:
        config = ColorConfig()

    # Concatenate all samples from this shot
    all_sdr = np.concatenate(sdr_samples) if sdr_samples else np.array([])
    all_hdr = np.concatenate(hdr_samples) if hdr_samples else np.array([])

    if len(all_sdr) < 100:
        logger.warning(f"Shot {shot.shot_id}: too few samples ({len(all_sdr)})")
        transform = ShotTransform(
            shot_id=shot.shot_id,
            luminance_curve=[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            confidence=0.0,
        )
        return transform

    # Estimate the curve
    curve = estimate_luminance_curve(all_sdr, all_hdr, config=config)

    # Estimate simple exposure/contrast as summary statistics
    # Exposure: ratio of medians
    sdr_median = float(np.median(all_sdr[all_sdr > 0.01]))
    hdr_median = float(np.median(all_hdr[all_hdr > 0.01])) if np.any(all_hdr > 0.01) else sdr_median
    exposure = hdr_median / sdr_median if sdr_median > 0 else 1.0

    # Contrast: ratio of IQR
    sdr_iqr = float(np.percentile(all_sdr, 75) - np.percentile(all_sdr, 25))
    hdr_iqr = float(np.percentile(all_hdr, 75) - np.percentile(all_hdr, 25))
    contrast = hdr_iqr / sdr_iqr if sdr_iqr > 0 else 1.0

    # Confidence: correlation between mapped SDR and actual HDR
    mapped = apply_luminance_curve(all_sdr, curve)
    if len(mapped) > 0 and np.std(mapped) > 0 and np.std(all_hdr) > 0:
        correlation = float(np.corrcoef(mapped.flatten(), all_hdr.flatten())[0, 1])
        confidence = max(0.0, correlation)
    else:
        confidence = 0.0

    transform = ShotTransform(
        shot_id=shot.shot_id,
        luminance_curve=curve,
        exposure=exposure,
        contrast=contrast,
        confidence=confidence,
    )

    logger.info(
        f"Shot {shot.shot_id}: exposure={exposure:.3f}, contrast={contrast:.3f}, "
        f"confidence={confidence:.4f}"
    )
    return transform
