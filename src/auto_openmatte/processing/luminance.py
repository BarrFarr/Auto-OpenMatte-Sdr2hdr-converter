"""Luminance mapping estimation.

Derives the SDR → HDR luminance transfer function from the overlap region.
Produces a monotonic mapping that can be applied to the extension areas.

Key principles:
- HDR is MASTER — we map FROM SDR TO HDR, never the reverse
- Mapping must be monotonic (no inversions)
- Uses robust statistics (percentile bins, outlier rejection)
- Fitted in LOG DOMAIN for better dynamic range coverage
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

# Peak luminance for normalization (cd/m²)
_PEAK_NITS = 10000.0

# Log-domain epsilon (stable across 1e-7 to 1e-4 per P2.5 validation)
_LOG_EPS = 1e-6


def estimate_luminance_curve(
    sdr_luminance: NDArray[np.floating],
    hdr_luminance: NDArray[np.floating],
    config: ColorConfig | None = None,
) -> list[list[float]]:
    """Estimate the SDR → HDR luminance mapping curve in log domain.

    Fits in log10(nits + ε) space for better dynamic range coverage,
    then stores control points in log domain. Application converts
    back to linear.

    Args:
        sdr_luminance: Flattened SDR luminance values (linear, [0, 1]).
        hdr_luminance: Corresponding HDR luminance values (linear, normalized by peak_nits).
        config: Color configuration.

    Returns:
        List of [sdr_log, hdr_log] control points in log10(nits+ε) domain.
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
        # Return identity-like curve in log domain
        return [[-6.0, -6.0], [0.0, 0.0], [4.0, 4.0]]

    # Step 2: Transform to log domain (nits scale)
    sdr_log = np.log10(sdr_clean * _PEAK_NITS + _LOG_EPS)
    hdr_log = np.log10(hdr_clean * _PEAK_NITS + _LOG_EPS)

    # Step 3: Percentile clipping and binning in log domain
    percentile_range = (config.low_percentile, config.high_percentile)

    bin_centers, bin_medians = percentile_bins(
        sdr_log, hdr_log,
        n_bins=config.luminance_bins,
        percentile_range=percentile_range,
    )

    if len(bin_centers) < 5:
        logger.warning("Too few valid bins for luminance curve")
        return [[-6.0, -6.0], [0.0, 0.0], [4.0, 4.0]]

    # Step 4: Enforce monotonicity
    x_mono, y_mono = fit_monotonic_spline(bin_centers, bin_medians)

    # Step 5: Subsample to 64 control points
    n_points = min(64, len(x_mono))
    indices = np.linspace(0, len(x_mono) - 1, n_points, dtype=int)
    x_final = x_mono[indices]
    y_final = y_mono[indices]

    # Build control point list (in log domain)
    curve = [[float(x), float(y)] for x, y in zip(x_final, y_final)]

    logger.info(
        f"Luminance curve (log domain): {len(curve)} control points, "
        f"SDR log range [{curve[0][0]:.3f}, {curve[-1][0]:.3f}] -> "
        f"HDR log range [{curve[0][1]:.3f}, {curve[-1][1]:.3f}]"
    )
    return curve


# ---------------------------------------------------------------------------
# LUT infrastructure (P2.15)
# ---------------------------------------------------------------------------

_LUT_SIZE = 65536  # 65k entries — validated in P2.14


def build_curve_lut(
    curve: list[list[float]],
    n_entries: int = _LUT_SIZE,
) -> dict:
    """Build a precomputed LUT for the FITTED range of a luminance curve.

    The LUT replaces the expensive log10 → PCHIP → pow10 path with a
    single array lookup.  It covers ONLY [curve_start_linear, curve_max_linear].
    Black and low-end bridge are NOT in the LUT.

    Each fitted curve (i.e. each shot) gets its own LUT.

    Args:
        curve: Control points [[sdr_log, hdr_log], ...] in log domain.
        n_entries: Number of LUT entries (default 65536).

    Returns:
        Dictionary with:
            - lut: ndarray of mapped HDR values (float64, n_entries)
            - curve_start_linear: start of fitted range (normalized)
            - curve_max_linear: end of fitted range (normalized)
            - curve_start_hdr_linear: HDR value at curve start (for bridge)
            - inv_range: 1/(curve_max_linear - curve_start_linear) for fast lookup
    """
    from scipy.interpolate import PchipInterpolator

    x_points = np.array([p[0] for p in curve])
    y_points = np.array([p[1] for p in curve])

    curve_x_min = x_points[0]
    curve_x_max = x_points[-1]

    # Linear domain bounds
    curve_start_nits = 10.0**curve_x_min - _LOG_EPS
    curve_start_linear = curve_start_nits / _PEAK_NITS
    curve_max_nits = 10.0**curve_x_max - _LOG_EPS
    curve_max_linear = curve_max_nits / _PEAK_NITS
    curve_start_hdr_nits = 10.0**y_points[0] - _LOG_EPS
    curve_start_hdr_linear = curve_start_hdr_nits / _PEAK_NITS

    # Build LUT: uniformly sample [curve_start_linear, curve_max_linear]
    lut_input = np.linspace(curve_start_linear, curve_max_linear, n_entries)

    # Compute exact PCHIP output for each sample
    sdr_log = np.log10(lut_input * _PEAK_NITS + _LOG_EPS)
    sdr_log_clamped = np.minimum(sdr_log, curve_x_max)
    interpolator = PchipInterpolator(x_points, y_points, extrapolate=False)
    hdr_log = interpolator(sdr_log_clamped)
    hdr_log = np.nan_to_num(hdr_log, nan=y_points[0])
    hdr_nits = np.power(10.0, hdr_log) - _LOG_EPS
    lut_output = np.maximum(hdr_nits, 0.0) / _PEAK_NITS

    inv_range = 1.0 / (curve_max_linear - curve_start_linear)

    return {
        "lut": lut_output,
        "curve_start_linear": curve_start_linear,
        "curve_max_linear": curve_max_linear,
        "curve_start_hdr_linear": curve_start_hdr_linear,
        "inv_range": inv_range,
    }


def apply_luminance_curve(
    sdr_luminance: NDArray[np.floating],
    curve: list[list[float]],
    *,
    prebuilt_lut: dict | None = None,
) -> NDArray[np.floating]:
    """Apply a luminance mapping curve to SDR values (production LUT path).

    The curve is stored in log10(nits+ε) domain. This function handles
    four ranges:
    1. TRUE BLACK (sdr <= 0): output = 0
    2. LOW-END BRIDGE (0 < sdr < curve_start): linear interpolation
       from (0, 0) to (curve_start_linear, curve_start_hdr) — no extrapolation
    3. FITTED RANGE (curve_start <= sdr <= curve_max): LUT 65k lookup
    4. TOP-END (sdr > curve_max): clamped to curve max output

    If prebuilt_lut is None, the LUT is built on first call (then should
    be cached externally for subsequent frames of the same shot).

    Args:
        sdr_luminance: Input SDR luminance values (linear, [0, 1]).
        curve: Control points [[sdr_log, hdr_log], ...] in log domain.
        prebuilt_lut: Optional pre-built LUT dict from build_curve_lut().

    Returns:
        Mapped HDR luminance values (linear, normalized [0,1]).
    """
    if not curve or len(curve) < 2:
        return sdr_luminance.copy()

    # Build or reuse LUT
    if prebuilt_lut is None:
        prebuilt_lut = build_curve_lut(curve)

    lut = prebuilt_lut["lut"]
    curve_start_linear = prebuilt_lut["curve_start_linear"]
    curve_start_hdr_linear = prebuilt_lut["curve_start_hdr_linear"]
    inv_range = prebuilt_lut["inv_range"]
    n_entries = len(lut)

    # Initialize output
    result = np.zeros_like(sdr_luminance)

    # === BRANCH A: TRUE BLACK (sdr <= 0) → 0 ===
    # (already initialized to 0)

    # === BRANCH B: LOW-END BRIDGE (0 < sdr < curve_start_linear) ===
    bridge_mask = (sdr_luminance > 0) & (sdr_luminance < curve_start_linear)
    if np.any(bridge_mask):
        bridge_lum = sdr_luminance[bridge_mask]
        result[bridge_mask] = (bridge_lum / curve_start_linear) * curve_start_hdr_linear

    # === BRANCH C: FITTED RANGE (sdr >= curve_start_linear) via LUT ===
    fitted_mask = sdr_luminance >= curve_start_linear
    if np.any(fitted_mask):
        fitted_lum = sdr_luminance[fitted_mask]
        # Normalize to [0, 1] within LUT range
        t = (fitted_lum - curve_start_linear) * inv_range
        t = np.clip(t, 0.0, 1.0)
        # Integer index into LUT
        idx = (t * (n_entries - 1)).astype(np.int64)
        idx = np.clip(idx, 0, n_entries - 1)
        result[fitted_mask] = lut[idx]

    return result


def apply_luminance_curve_reference(
    sdr_luminance: NDArray[np.floating],
    curve: list[list[float]],
) -> NDArray[np.floating]:
    """Reference PCHIP implementation for regression testing.

    This is the original P2.9.3 implementation preserved for numerical
    validation of the LUT production path.  Not used in normal renders.

    Args:
        sdr_luminance: Input SDR luminance values (linear, [0, 1]).
        curve: Control points [[sdr_log, hdr_log], ...] in log domain.

    Returns:
        Mapped HDR luminance values (linear, normalized [0,1]).
    """
    if not curve or len(curve) < 2:
        return sdr_luminance.copy()

    from scipy.interpolate import PchipInterpolator

    x_points = np.array([p[0] for p in curve])
    y_points = np.array([p[1] for p in curve])

    # Curve bounds in log domain
    curve_x_min = x_points[0]
    curve_x_max = x_points[-1]

    # Convert curve start to linear normalized units
    curve_start_nits = 10.0**curve_x_min - _LOG_EPS
    curve_start_linear = curve_start_nits / _PEAK_NITS

    # Curve start HDR value
    curve_start_hdr_nits = 10.0**y_points[0] - _LOG_EPS
    curve_start_hdr_linear = curve_start_hdr_nits / _PEAK_NITS

    # Initialize output
    result = np.zeros_like(sdr_luminance)

    # === BRANCH A: TRUE BLACK (sdr <= 0) → 0 ===
    # (already initialized to 0)

    # === BRANCH B: LOW-END BRIDGE (0 < sdr < curve_start_linear) ===
    bridge_mask = (sdr_luminance > 0) & (sdr_luminance < curve_start_linear)
    if np.any(bridge_mask):
        bridge_lum = sdr_luminance[bridge_mask]
        result[bridge_mask] = (bridge_lum / curve_start_linear) * curve_start_hdr_linear

    # === BRANCH C: FITTED RANGE (sdr >= curve_start_linear) ===
    fitted_mask = sdr_luminance >= curve_start_linear
    if np.any(fitted_mask):
        fitted_lum = sdr_luminance[fitted_mask]
        sdr_log = np.log10(fitted_lum * _PEAK_NITS + _LOG_EPS)

        # Clamp upper end only (no extrapolation above curve)
        sdr_log_clamped = np.minimum(sdr_log, curve_x_max)

        # PCHIP interpolation within fitted range
        interpolator = PchipInterpolator(x_points, y_points, extrapolate=False)
        hdr_log = interpolator(sdr_log_clamped)

        # Handle NaN edge cases
        hdr_log = np.nan_to_num(hdr_log, nan=y_points[0])

        # Convert back to linear normalized
        hdr_nits = np.power(10.0, hdr_log) - _LOG_EPS
        result[fitted_mask] = np.maximum(hdr_nits, 0.0) / _PEAK_NITS

    return result


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
