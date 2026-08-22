"""Color space utilities for SDR-to-HDR reshaping research.

Provides BT.709 gamma encode/decode, PQ EOTF/OETF, BT.709/BT.2020 gamut
conversion, and YCbCr decomposition. All functions operate on numpy arrays
with shape (..., 3) for RGB or (...,) for single-channel values in [0, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# PQ (ST 2084) constants
# =============================================================================
_PQ_M1: float = 2610.0 / 16384.0
_PQ_M2: float = 2523.0 / 4096.0 * 128.0
_PQ_C1: float = 3424.0 / 4096.0
_PQ_C2: float = 2413.0 / 4096.0 * 32.0
_PQ_C3: float = 2392.0 / 4096.0 * 32.0
_PQ_PEAK_LUMINANCE: float = 10000.0

# =============================================================================
# BT.709 <-> BT.2020 conversion matrices
# =============================================================================
_BT709_TO_BT2020: NDArray[np.floating] = np.array(
    [
        [0.6274040, 0.3292820, 0.0433136],
        [0.0690970, 0.9195400, 0.0113612],
        [0.0163916, 0.0880132, 0.8955950],
    ],
    dtype=np.float64,
)

_BT2020_TO_BT709: NDArray[np.floating] = np.array(
    [
        [1.6604910, -0.5876411, -0.0728499],
        [-0.1245505, 1.1328999, -0.0083494],
        [-0.0181508, -0.1005789, 1.1187297],
    ],
    dtype=np.float64,
)

# BT.709 luminance weights
_BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# BT.2020 luminance weights
_BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


# =============================================================================
# SDR gamma (BT.709 / BT.1886 gamma 2.4)
# =============================================================================

def sdr_gamma_to_linear(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Decode BT.709 gamma 2.4 signal to linear light.

    Args:
        img: Gamma-encoded values in [0, 1].

    Returns:
        Linear light values in [0, 1].
    """
    img = np.asarray(img, dtype=np.float64)
    return np.power(np.clip(img, 0.0, 1.0), 2.4)


def linear_to_sdr_gamma(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Encode linear light to BT.709 gamma 2.4 signal.

    Args:
        img: Linear light values in [0, 1].

    Returns:
        Gamma-encoded values in [0, 1].
    """
    img = np.asarray(img, dtype=np.float64)
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.4)


# =============================================================================
# PQ EOTF / OETF
# =============================================================================

def pq_eotf(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """PQ EOTF: decode PQ code values [0,1] to linear light (nits/10000).

    Args:
        img: PQ-encoded signal values in [0, 1].

    Returns:
        Linear light normalized to [0, 1] (where 1.0 = 10000 nits).
    """
    img = np.asarray(img, dtype=np.float64)
    signal = np.clip(img, 0.0, 1.0)
    vp = np.power(signal, 1.0 / _PQ_M2)
    num = np.maximum(vp - _PQ_C1, 0.0)
    den = np.maximum(_PQ_C2 - _PQ_C3 * vp, 1e-12)
    linear = np.power(num / den, 1.0 / _PQ_M1)
    return linear


def pq_oetf(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """PQ OETF: encode linear light (nits/10000) to PQ code values [0,1].

    Args:
        img: Linear light normalized to [0, 1] (where 1.0 = 10000 nits).

    Returns:
        PQ-encoded signal values in [0, 1].
    """
    img = np.asarray(img, dtype=np.float64)
    y = np.clip(img, 0.0, 1.0)
    ym1 = np.power(y, _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * ym1
    den = 1.0 + _PQ_C3 * ym1
    signal = np.power(num / den, _PQ_M2)
    return signal


# =============================================================================
# BT.709 <-> BT.2020 gamut conversion
# =============================================================================

def bt709_to_bt2020(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert linear BT.709 RGB to linear BT.2020 RGB.

    Args:
        img: Linear BT.709 RGB of shape (..., 3).

    Returns:
        Linear BT.2020 RGB of same shape.
    """
    img = np.asarray(img, dtype=np.float64)
    return np.einsum("ij,...j->...i", _BT709_TO_BT2020, img)


def bt2020_to_bt709(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert linear BT.2020 RGB to linear BT.709 RGB.

    Args:
        img: Linear BT.2020 RGB of shape (..., 3).

    Returns:
        Linear BT.709 RGB of same shape.
    """
    img = np.asarray(img, dtype=np.float64)
    return np.einsum("ij,...j->...i", _BT2020_TO_BT709, img)


# =============================================================================
# YCbCr decomposition (based on BT.709 weights for SDR content)
# =============================================================================

def rgb_to_ycbcr(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Decompose linear RGB into luminance (Y) and chroma (Cb, Cr).

    Uses BT.709 luminance weights. The result is:
      Y  = 0.2126*R + 0.7152*G + 0.0722*B
      Cb = (B - Y) / 1.8556
      Cr = (R - Y) / 1.5748

    Args:
        img: Linear RGB of shape (..., 3).

    Returns:
        YCbCr array of shape (..., 3).
    """
    img = np.asarray(img, dtype=np.float64)
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cb = (B - Y) / 1.8556
    Cr = (R - Y) / 1.5748

    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr_to_rgb(img: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert YCbCr back to linear RGB.

    Inverse of rgb_to_ycbcr.

    Args:
        img: YCbCr array of shape (..., 3).

    Returns:
        Linear RGB of shape (..., 3).
    """
    img = np.asarray(img, dtype=np.float64)
    Y = img[..., 0]
    Cb = img[..., 1]
    Cr = img[..., 2]

    R = Y + 1.5748 * Cr
    B = Y + 1.8556 * Cb
    G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152

    return np.stack([R, G, B], axis=-1)
