"""Quality metrics for SDR-to-HDR reshaping evaluation.

All metrics operate on arrays in linear BT.2020 space [0, 1] representing
0-10000 nits unless otherwise specified.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

# BT.2020 luminance weights
_BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
_PEAK_NITS = 10000.0


def _to_nits(img):
    """Convert normalized [0, 1] to nits."""
    return np.asarray(img, dtype=np.float64) * _PEAK_NITS


def _luminance_nits(img):
    """Extract luminance in nits from linear BT.2020 RGB [0, 1]."""
    return np.einsum("...c,c->...", img, _BT2020_LUMA) * _PEAK_NITS


# =============================================================================
# Luminance metrics
# =============================================================================

def luminance_mae(pred, ref):
    """Mean absolute error of luminance in nits.

    Args:
        pred: Predicted HDR image, linear BT.2020 [0, 1], shape (H, W, 3).
        ref: Reference HDR image, same format.

    Returns:
        MAE in nits.
    """
    pred_lum = _luminance_nits(pred)
    ref_lum = _luminance_nits(ref)
    return float(np.mean(np.abs(pred_lum - ref_lum)))


def luminance_rmse(pred, ref):
    """Root mean squared error of luminance in nits.

    Args:
        pred: Predicted HDR image, linear BT.2020 [0, 1], shape (H, W, 3).
        ref: Reference HDR image, same format.

    Returns:
        RMSE in nits.
    """
    pred_lum = _luminance_nits(pred)
    ref_lum = _luminance_nits(ref)
    return float(np.sqrt(np.mean((pred_lum - ref_lum)**2)))


# =============================================================================
# Color metric: approximate CIE deltaE 2000
# =============================================================================

def delta_e_2000(pred, ref):
    """Approximate CIE deltaE2000 between predicted and reference.

    Converts both to L*a*b* and computes mean deltaE2000.

    Args:
        pred: Predicted image, linear BT.2020 [0, 1], shape (H, W, 3).
        ref: Reference image, same format.

    Returns:
        Mean deltaE2000 value.
    """
    pred_lab = _linear_to_lab(pred)
    ref_lab = _linear_to_lab(ref)

    p = pred_lab.reshape(-1, 3)
    r = ref_lab.reshape(-1, 3)

    de = _delta_e_2000_batch(p, r)
    return float(np.mean(de))


def _linear_to_lab(img):
    """Convert linear BT.2020 RGB [0, 1] to CIE L*a*b* (approximate)."""
    rgb_to_xyz = np.array(
        [[0.6370, 0.1446, 0.1689],
         [0.2627, 0.6780, 0.0593],
         [0.0000, 0.0281, 1.0610]],
        dtype=np.float64,
    )
    xyz = np.einsum("ij,...j->...i", rgb_to_xyz, img)

    d65 = np.array([0.9505, 1.0000, 1.0890], dtype=np.float64)
    xyz_n = xyz / d65

    delta = 6.0 / 29.0
    delta_sq = delta**2
    delta_cb = delta**3

    f = np.where(
        xyz_n > delta_cb,
        np.cbrt(np.maximum(xyz_n, 0.0)),
        xyz_n / (3.0 * delta_sq) + 4.0 / 29.0,
    )

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def _delta_e_2000_batch(lab1, lab2):
    """Compute deltaE2000 for batched L*a*b* pairs (simplified)."""
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cab = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt(Cab**7 / (Cab**7 + 25.0**7)))
    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.zeros_like(h1p)
    mask1 = np.abs(h2p - h1p) <= 180.0
    dhp[mask1] = (h2p - h1p)[mask1]
    mask2 = (~mask1) & (h2p > h1p)
    dhp[mask2] = (h2p - h1p - 360.0)[mask2]
    mask3 = (~mask1) & (h2p <= h1p)
    dhp[mask3] = (h2p - h1p + 360.0)[mask3]

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))

    Lp = (L1 + L2) / 2.0
    Cp = (C1p + C2p) / 2.0

    hp = np.zeros_like(h1p)
    abs_diff = np.abs(h1p - h2p)
    sum_h = h1p + h2p
    cond1 = abs_diff <= 180.0
    hp[cond1] = sum_h[cond1] / 2.0
    cond2 = (~cond1) & (sum_h < 360.0)
    hp[cond2] = (sum_h[cond2] + 360.0) / 2.0
    cond3 = (~cond1) & (sum_h >= 360.0)
    hp[cond3] = (sum_h[cond3] - 360.0) / 2.0

    T = (1.0
         - 0.17 * np.cos(np.radians(hp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp))
         + 0.32 * np.cos(np.radians(3.0 * hp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp - 63.0)))

    SL = 1.0 + 0.015 * (Lp - 50.0)**2 / np.sqrt(20.0 + (Lp - 50.0)**2)
    SC = 1.0 + 0.045 * Cp
    SH = 1.0 + 0.015 * Cp * T

    RT_term = -np.sin(2.0 * np.radians(60.0 * np.exp(-((hp - 275.0) / 25.0)**2)))
    RC = 2.0 * np.sqrt(Cp**7 / (Cp**7 + 25.0**7))
    RT = RT_term * RC

    dE = np.sqrt(
        (dLp / SL)**2
        + (dCp / SC)**2
        + (dHp / SH)**2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return dE


# =============================================================================
# Seam (boundary) error
# =============================================================================

def seam_error(full_image, boundary_row, width=10):
    """Measure gradient discontinuity at the HDR/extension border.

    Args:
        full_image: Full open matte result image, shape (H, W, 3).
        boundary_row: Row index of the seam boundary.
        width: Number of rows above and below boundary to analyze.

    Returns:
        Seam error score (0 = perfect continuity, higher = worse).
    """
    H = full_image.shape[0]
    if boundary_row < width or boundary_row >= H - width:
        return 0.0

    lum = np.einsum("...c,c->...", full_image, _BT2020_LUMA)

    boundary_grad = np.abs(lum[boundary_row, :] - lum[boundary_row - 1, :])

    above = lum[boundary_row - width:boundary_row - 1, :]
    below = lum[boundary_row + 1:boundary_row + width, :]

    above_grads = np.abs(np.diff(above, axis=0))
    below_grads = np.abs(np.diff(below, axis=0))

    if above_grads.size == 0 or below_grads.size == 0:
        return 0.0

    avg_surrounding_grad = float(np.mean(np.concatenate([above_grads, below_grads])))
    boundary_mean = float(np.mean(boundary_grad))
    seam_err = max(0.0, boundary_mean - avg_surrounding_grad)

    return seam_err * _PEAK_NITS


# =============================================================================
# Temporal stability
# =============================================================================

def temporal_stability(params_list):
    """Measure parameter variance across multiple fits (simulating frames).

    Args:
        params_list: List of parameter arrays (one per simulated frame).

    Returns:
        Mean normalized standard deviation across parameters.
    """
    if not params_list or len(params_list) < 2:
        return 0.0

    try:
        stacked = np.array(params_list, dtype=np.float64)
    except (ValueError, TypeError):
        return 0.0

    if stacked.ndim != 2:
        return 0.0

    std = np.std(stacked, axis=0)
    mean = np.abs(np.mean(stacked, axis=0))

    valid = mean > 1e-10
    if not np.any(valid):
        return 0.0

    cv = std[valid] / mean[valid]
    return float(np.mean(cv))
