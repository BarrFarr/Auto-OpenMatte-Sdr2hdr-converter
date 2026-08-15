"""Mathematical utilities — robust statistics, spline fitting, outlier rejection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def robust_median(values: NDArray[np.floating]) -> float:
    """Compute median, ignoring NaN."""
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return 0.0
    return float(np.median(clean))


def iqr_filter(
    values: NDArray[np.floating], multiplier: float = 1.5
) -> NDArray[np.bool_]:
    """Return boolean mask of inliers using IQR filtering.

    Args:
        values: 1D array of values.
        multiplier: IQR multiplier for fence.

    Returns:
        Boolean mask (True = inlier).
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (values >= lower) & (values <= upper)


def percentile_bins(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    n_bins: int = 256,
    percentile_range: tuple[float, float] = (1.0, 99.0),
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Bin x values and compute robust y median per bin.

    Used for luminance curve estimation.

    Args:
        x: Input values (e.g., SDR luminance).
        y: Target values (e.g., HDR luminance).
        n_bins: Number of bins.
        percentile_range: Range of x to consider.

    Returns:
        Tuple of (bin_centers, bin_medians) for the valid bins.
    """
    x_min = np.percentile(x, percentile_range[0])
    x_max = np.percentile(x, percentile_range[1])

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_medians = np.zeros(n_bins)
    valid = np.zeros(n_bins, dtype=bool)

    for i in range(n_bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if np.sum(mask) >= 10:  # Minimum samples per bin
            bin_medians[i] = np.median(y[mask])
            valid[i] = True

    return bin_centers[valid], bin_medians[valid]


def fit_monotonic_spline(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Fit a monotonically increasing spline through data points.

    Uses isotonic regression to enforce monotonicity, then returns
    the corrected points suitable for PCHIP interpolation.

    Args:
        x: Sorted x values (must be strictly increasing).
        y: Corresponding y values.

    Returns:
        Tuple of (x, y_monotonic) where y_monotonic is guaranteed non-decreasing.
    """
    # Enforce monotonicity via pool adjacent violators (isotonic regression)
    y_mono = np.copy(y)
    n = len(y_mono)

    # Forward pass: ensure non-decreasing
    for i in range(1, n):
        if y_mono[i] < y_mono[i - 1]:
            # Average with previous to maintain overall shape
            avg = (y_mono[i] + y_mono[i - 1]) / 2.0
            y_mono[i] = avg
            y_mono[i - 1] = avg

    # Backward pass for smoothness
    for i in range(n - 2, -1, -1):
        if y_mono[i] > y_mono[i + 1]:
            avg = (y_mono[i] + y_mono[i + 1]) / 2.0
            y_mono[i] = avg
            y_mono[i + 1] = avg

    # Final enforcement: strictly non-decreasing
    for i in range(1, n):
        y_mono[i] = max(y_mono[i], y_mono[i - 1])

    return x, y_mono


def estimate_3x3_matrix(
    src: NDArray[np.floating], dst: NDArray[np.floating], regularize: float = 0.01
) -> NDArray[np.floating]:
    """Estimate a 3x3 color correction matrix using least squares.

    Solves: dst = src @ M.T (per-pixel, in linear RGB)
    with regularization toward the identity matrix.

    Args:
        src: Source pixels, shape (N, 3).
        dst: Destination pixels, shape (N, 3).
        regularize: Regularization strength toward identity.

    Returns:
        3x3 color matrix.
    """
    n = src.shape[0]
    # Build regularized system: minimize ||src @ M.T - dst||^2 + lambda * ||M - I||^2
    # Solve per output channel
    matrix = np.eye(3, dtype=np.float64)
    reg_matrix = regularize * np.eye(3)

    for ch in range(3):
        # A @ m = b  where A = src, b = dst[:, ch]
        ata = src.T @ src + n * reg_matrix
        atb = src.T @ dst[:, ch] + n * regularize * np.eye(3)[ch]
        try:
            matrix[ch] = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            matrix[ch] = np.eye(3)[ch]  # Fallback to identity for this channel

    return matrix


def normalized_cross_correlation(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> float:
    """Compute normalized cross-correlation between two arrays.

    Args:
        a: First array.
        b: Second array (must be same shape as a).

    Returns:
        NCC value in [-1, 1], where 1 = perfect match.
    """
    a_norm = a - np.mean(a)
    b_norm = b - np.mean(b)
    denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a_norm * b_norm) / denom)
