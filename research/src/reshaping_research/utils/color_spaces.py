"""Color space utilities - BT.709/BT.2020 matrices, luminance weights, gamut helpers.

All matrices and operations follow ITU-R recommendations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ============================================================
# Luminance coefficients
# ============================================================
# BT.709 luminance weights (Y = 0.2126R + 0.7152G + 0.0722B)
BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# BT.2020 luminance weights (Y = 0.2627R + 0.6780G + 0.0593B)
BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)

# ============================================================
# Color space conversion matrices
# ============================================================
# BT.709 -> BT.2020 linear RGB conversion matrix
# Derived from chromaticity coordinates of both standards
BT709_TO_BT2020: NDArray[np.floating] = np.array(
    [
        [0.6274040, 0.3292820, 0.0433136],
        [0.0690970, 0.9195400, 0.0113612],
        [0.0163916, 0.0880132, 0.8955950],
    ],
    dtype=np.float64,
)

# BT.2020 -> BT.709 linear RGB conversion matrix (inverse of above)
BT2020_TO_BT709: NDArray[np.floating] = np.array(
    [
        [1.6604910, -0.5876411, -0.0728499],
        [-0.1245505, 1.1328999, -0.0083494],
        [-0.0181508, -0.1005789, 1.1187297],
    ],
    dtype=np.float64,
)

# ============================================================
# ICtCp conversion matrices (via LMS)
# ============================================================
# BT.2020 RGB -> LMS (for ICtCp)
BT2020_TO_LMS: NDArray[np.floating] = np.array(
    [
        [1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0],
        [683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0],
        [99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0],
    ],
    dtype=np.float64,
)

# LMS -> ICtCp rotation
LMS_TO_ICTCP: NDArray[np.floating] = np.array(
    [
        [2048.0 / 4096.0, 2048.0 / 4096.0, 0.0],
        [6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0],
        [17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0],
    ],
    dtype=np.float64,
)

# ICtCp -> LMS (inverse)
ICTCP_TO_LMS: NDArray[np.floating] = np.linalg.inv(LMS_TO_ICTCP)

# LMS -> BT.2020 RGB (inverse)
LMS_TO_BT2020: NDArray[np.floating] = np.linalg.inv(BT2020_TO_LMS)


def compute_luminance(
    rgb: NDArray[np.floating], weights: NDArray[np.floating] = BT2020_LUMA
) -> NDArray[np.floating]:
    """Compute luminance (Y) from linear RGB using given luma weights.

    Args:
        rgb: Linear RGB image array of shape (..., 3).
        weights: Luminance weight coefficients [R, G, B].

    Returns:
        Luminance array of shape (...).
    """
    return np.einsum("...c,c->...", rgb, weights)


def bt709_to_bt2020(rgb709: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert linear BT.709 RGB to linear BT.2020 RGB.

    Args:
        rgb709: Linear BT.709 RGB image of shape (..., 3).

    Returns:
        Linear BT.2020 RGB image of same shape.
    """
    return np.einsum("ij,...j->...i", BT709_TO_BT2020, rgb709)


def bt2020_to_bt709(rgb2020: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert linear BT.2020 RGB to linear BT.709 RGB.

    Args:
        rgb2020: Linear BT.2020 RGB image of shape (..., 3).

    Returns:
        Linear BT.709 RGB image of same shape.
    """
    return np.einsum("ij,...j->...i", BT2020_TO_BT709, rgb2020)


def gamut_clip_bt709(rgb709: NDArray[np.floating]) -> NDArray[np.floating]:
    """Simple gamut clip: clamp linear BT.709 RGB to [0, 1].

    Args:
        rgb709: Linear BT.709 RGB values.

    Returns:
        Clipped values in [0, 1].
    """
    return np.clip(rgb709, 0.0, 1.0)


def linear_bt2020_to_ictcp(
    rgb2020: NDArray[np.floating], peak_nits: float = 10000.0
) -> NDArray[np.floating]:
    """Convert linear BT.2020 RGB (in nits) to ICtCp.

    Args:
        rgb2020: Linear BT.2020 RGB in absolute luminance (nits), shape (..., 3).
        peak_nits: Peak luminance for PQ encoding.

    Returns:
        ICtCp values, shape (..., 3). I is intensity, Ct is tritan, Cp is protan.
    """
    from .transfer_functions import pq_oetf

    # RGB -> LMS
    lms = np.einsum("ij,...j->...i", BT2020_TO_LMS, rgb2020)
    lms = np.maximum(lms, 0.0)

    # Apply PQ to LMS
    lms_pq = pq_oetf(lms)

    # LMS_PQ -> ICtCp
    ictcp = np.einsum("ij,...j->...i", LMS_TO_ICTCP, lms_pq)
    return ictcp


def ictcp_to_linear_bt2020(
    ictcp: NDArray[np.floating], peak_nits: float = 10000.0
) -> NDArray[np.floating]:
    """Convert ICtCp to linear BT.2020 RGB (in nits).

    Args:
        ictcp: ICtCp values, shape (..., 3).
        peak_nits: Peak luminance for PQ decoding.

    Returns:
        Linear BT.2020 RGB in absolute luminance (nits), shape (..., 3).
    """
    from .transfer_functions import pq_eotf

    # ICtCp -> LMS_PQ
    lms_pq = np.einsum("ij,...j->...i", ICTCP_TO_LMS, ictcp)

    # Inverse PQ -> linear LMS
    lms = pq_eotf(lms_pq)

    # LMS -> BT.2020 RGB
    rgb2020 = np.einsum("ij,...j->...i", LMS_TO_BT2020, lms)
    return rgb2020


def linear_to_lab_d65(linear_rgb: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert linear sRGB/BT.709 RGB to CIE L*a*b* under D65.

    Simplified implementation for deltaE calculations.

    Args:
        linear_rgb: Linear RGB values in [0, 1], shape (..., 3).

    Returns:
        L*a*b* values, shape (..., 3).
    """
    # sRGB/BT.709 to XYZ (D65 adapted)
    rgb_to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = np.einsum("ij,...j->...i", rgb_to_xyz, linear_rgb)

    # D65 white point
    d65_white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_norm = xyz / d65_white

    # f(t) function for Lab
    delta = 6.0 / 29.0
    delta_sq = delta * delta
    delta_cb = delta * delta_sq

    f = np.where(
        xyz_norm > delta_cb,
        np.cbrt(xyz_norm),
        xyz_norm / (3.0 * delta_sq) + 4.0 / 29.0,
    )

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)
