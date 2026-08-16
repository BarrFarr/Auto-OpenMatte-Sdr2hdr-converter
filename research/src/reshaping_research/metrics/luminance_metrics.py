"""Luminance metrics: MAE, RMSE, percentile error, segmented errors.

All metrics operate on linear luminance in nits (cd/m^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentedErrors:
    """Luminance errors segmented by tonal range."""

    shadow_mae: float  # < 50 nits
    midtone_mae: float  # 50-200 nits
    highlight_mae: float  # > 200 nits
    shadow_rmse: float
    midtone_rmse: float
    highlight_rmse: float


def luminance_mae(
    predicted: NDArray[np.floating], reference: NDArray[np.floating]
) -> float:
    """Mean absolute error in linear nits.

    Args:
        predicted: Predicted luminance values in nits.
        reference: Reference (ground truth) luminance values in nits.

    Returns:
        MAE value in nits.
    """
    diff = np.abs(predicted.ravel() - reference.ravel())
    return float(np.mean(diff))


def luminance_rmse(
    predicted: NDArray[np.floating], reference: NDArray[np.floating]
) -> float:
    """Root mean squared error in linear nits.

    Args:
        predicted: Predicted luminance values in nits.
        reference: Reference (ground truth) luminance values in nits.

    Returns:
        RMSE value in nits.
    """
    diff = predicted.ravel() - reference.ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def percentile_errors(
    predicted: NDArray[np.floating],
    reference: NDArray[np.floating],
    percentiles: Tuple[float, ...] = (50.0, 90.0, 99.0),
) -> Dict[str, float]:
    """Percentile-based absolute errors.

    Computes the Pxx percentile of the absolute error distribution.

    Args:
        predicted: Predicted luminance values in nits.
        reference: Reference luminance values in nits.
        percentiles: Which percentiles to compute (default: P50, P90, P99).

    Returns:
        Dictionary mapping percentile labels (e.g., "P50") to error values in nits.
    """
    abs_errors = np.abs(predicted.ravel() - reference.ravel())
    results: Dict[str, float] = {}
    for p in percentiles:
        results[f"P{int(p)}"] = float(np.percentile(abs_errors, p))
    return results


def segmented_errors(
    predicted: NDArray[np.floating],
    reference: NDArray[np.floating],
    shadow_threshold: float = 50.0,
    highlight_threshold: float = 200.0,
) -> SegmentedErrors:
    """Compute errors segmented by luminance range (shadow/midtone/highlight).

    Segmentation is based on the reference luminance values.

    Args:
        predicted: Predicted luminance values in nits, shape matches reference.
        reference: Reference luminance values in nits.
        shadow_threshold: Upper bound for shadow region (default: 50 nits).
        highlight_threshold: Lower bound for highlight region (default: 200 nits).

    Returns:
        SegmentedErrors dataclass with MAE and RMSE for each region.
    """
    pred_flat = predicted.ravel()
    ref_flat = reference.ravel()

    shadow_mask = ref_flat < shadow_threshold
    mid_mask = (ref_flat >= shadow_threshold) & (ref_flat <= highlight_threshold)
    highlight_mask = ref_flat > highlight_threshold

    def _compute_segment(mask: NDArray[np.bool_]) -> Tuple[float, float]:
        if not np.any(mask):
            return 0.0, 0.0
        diff = pred_flat[mask] - ref_flat[mask]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        return mae, rmse

    s_mae, s_rmse = _compute_segment(shadow_mask)
    m_mae, m_rmse = _compute_segment(mid_mask)
    h_mae, h_rmse = _compute_segment(highlight_mask)

    return SegmentedErrors(
        shadow_mae=s_mae,
        midtone_mae=m_mae,
        highlight_mae=h_mae,
        shadow_rmse=s_rmse,
        midtone_rmse=m_rmse,
        highlight_rmse=h_rmse,
    )
