"""Color correction estimation.

After luminance mapping, estimates residual color differences
and produces a 3x3 correction matrix per shot.

Priority order:
1. Luminance (handled by luminance.py)
2. Neutral balance
3. Chroma
4. Saturation
5. 3x3 color correction matrix (gentle)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import ColorConfig
from auto_openmatte.utils.math_utils import estimate_3x3_matrix

logger = logging.getLogger(__name__)


def estimate_color_correction(
    sdr_rgb_linear: NDArray[np.floating],
    hdr_rgb_linear: NDArray[np.floating],
    config: ColorConfig | None = None,
) -> tuple[list[list[float]], float]:
    """Estimate a 3x3 color correction matrix from overlap samples.

    The matrix is applied AFTER luminance mapping to correct residual
    color shifts (white balance, gamut mapping, saturation).

    Uses regularized least squares with regularization toward identity
    to prevent aggressive corrections.

    Args:
        sdr_rgb_linear: SDR pixels in linear RGB, shape (N, 3).
            These should already have luminance mapping applied.
        hdr_rgb_linear: HDR pixels in linear RGB, shape (N, 3).
        config: Color configuration.

    Returns:
        Tuple of (3x3 matrix as nested list, saturation multiplier).
    """
    if config is None:
        config = ColorConfig()

    if len(sdr_rgb_linear) < 100 or len(hdr_rgb_linear) < 100:
        logger.warning("Too few samples for color correction")
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 1.0

    # Step 1: Outlier rejection
    # Remove pixels that are too dark or too bright (clipped)
    sdr_lum = np.mean(sdr_rgb_linear, axis=1)
    hdr_lum = np.mean(hdr_rgb_linear, axis=1)

    low_pct = config.low_percentile / 100.0
    high_pct = config.high_percentile / 100.0

    valid_mask = (
        (sdr_lum > np.quantile(sdr_lum, low_pct))
        & (sdr_lum < np.quantile(sdr_lum, high_pct))
        & (hdr_lum > np.quantile(hdr_lum, low_pct))
        & (hdr_lum < np.quantile(hdr_lum, high_pct))
        & np.all(np.isfinite(sdr_rgb_linear), axis=1)
        & np.all(np.isfinite(hdr_rgb_linear), axis=1)
    )

    sdr_clean = sdr_rgb_linear[valid_mask]
    hdr_clean = hdr_rgb_linear[valid_mask]

    if len(sdr_clean) < 50:
        logger.warning("Too few valid samples after outlier rejection")
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 1.0

    # Step 2: Subsample if too many pixels (for performance)
    max_samples = 100000
    if len(sdr_clean) > max_samples:
        indices = np.random.default_rng(42).choice(
            len(sdr_clean), max_samples, replace=False
        )
        sdr_clean = sdr_clean[indices]
        hdr_clean = hdr_clean[indices]

    # Step 3: Estimate 3x3 matrix with regularization toward identity
    # Higher regularization = more conservative correction
    matrix = estimate_3x3_matrix(sdr_clean, hdr_clean, regularize=0.05)

    # Step 4: Validate matrix
    # Check that it doesn't deviate too far from identity
    identity = np.eye(3)
    deviation = np.linalg.norm(matrix - identity, 'fro')

    if deviation > 1.0:
        logger.warning(
            f"Color matrix deviates significantly from identity ({deviation:.3f}). "
            "Reducing correction strength."
        )
        # Blend toward identity
        blend = 0.5
        matrix = blend * matrix + (1.0 - blend) * identity

    # Step 5: Estimate saturation multiplier
    # Compare chroma magnitude between SDR (transformed) and HDR
    sdr_chroma = np.sqrt(np.sum((sdr_clean - sdr_lum[valid_mask, np.newaxis]) ** 2, axis=1))
    hdr_chroma = np.sqrt(np.sum((hdr_clean - hdr_lum[valid_mask, np.newaxis]) ** 2, axis=1))

    # Use median ratio as saturation estimate
    sdr_chroma_med = np.median(sdr_chroma[sdr_chroma > 0.01])
    hdr_chroma_med = np.median(hdr_chroma[hdr_chroma > 0.01])

    if sdr_chroma_med > 0:
        saturation = float(hdr_chroma_med / sdr_chroma_med)
        # Clamp saturation to reasonable range
        saturation = max(0.5, min(2.0, saturation))
    else:
        saturation = 1.0

    # Confidence: how well does the matrix predict HDR from SDR?
    predicted = sdr_clean @ matrix.T
    residual = np.mean(np.abs(predicted - hdr_clean))
    target_range = np.mean(np.abs(hdr_clean))
    confidence = max(0.0, 1.0 - residual / (target_range + 1e-6))

    logger.info(
        f"Color correction: deviation={deviation:.4f}, saturation={saturation:.3f}, "
        f"confidence={confidence:.4f}"
    )

    matrix_list = matrix.tolist()
    return matrix_list, saturation


def apply_color_matrix(
    rgb_linear: NDArray[np.floating],
    matrix: list[list[float]],
) -> NDArray[np.floating]:
    """Apply a 3x3 color correction matrix to linear RGB pixels.

    Args:
        rgb_linear: Input pixels shape (..., 3) in linear RGB.
        matrix: 3x3 correction matrix.

    Returns:
        Corrected pixels, same shape. Clamped to non-negative.
    """
    mat = np.array(matrix, dtype=np.float64)
    original_shape = rgb_linear.shape
    pixels = rgb_linear.reshape(-1, 3)

    corrected = pixels @ mat.T
    corrected = np.maximum(corrected, 0.0)

    return corrected.reshape(original_shape)
