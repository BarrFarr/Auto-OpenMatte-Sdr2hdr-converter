"""Transfer function math — PQ (ST 2084), HLG, BT.1886.

All functions operate on numpy arrays for vectorized processing.
Values are normalized: signal domain [0, 1], luminance domain varies by standard.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ============================================================
# PQ (SMPTE ST 2084) — Perceptual Quantizer
# ============================================================
# Constants from SMPTE ST 2084
_PQ_M1 = 2610.0 / 16384.0  # 0.1593017578125
_PQ_M2 = 2523.0 / 4096.0 * 128.0  # 78.84375
_PQ_C1 = 3424.0 / 4096.0  # 0.8359375
_PQ_C2 = 2413.0 / 4096.0 * 32.0  # 18.8515625
_PQ_C3 = 2392.0 / 4096.0 * 32.0  # 18.6875
_PQ_PEAK_LUMINANCE = 10000.0  # cd/m²


def pq_eotf(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """PQ EOTF: signal [0,1] → linear luminance [0, 10000] cd/m².

    Converts PQ-encoded signal values to absolute linear luminance.
    """
    signal = np.clip(signal, 0.0, 1.0)
    # Raise to power 1/m2
    vp = np.power(signal, 1.0 / _PQ_M2)
    # Numerator and denominator
    num = np.maximum(vp - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * vp
    den = np.maximum(den, 1e-12)  # Avoid division by zero
    # Linear normalized
    linear = np.power(num / den, 1.0 / _PQ_M1)
    return linear * _PQ_PEAK_LUMINANCE


def pq_oetf(luminance: NDArray[np.floating]) -> NDArray[np.floating]:
    """PQ inverse EOTF (OETF): linear luminance [0, 10000] cd/m² → signal [0, 1].

    Converts absolute linear luminance to PQ-encoded signal values.
    """
    luminance = np.clip(luminance, 0.0, _PQ_PEAK_LUMINANCE)
    # Normalize to [0, 1]
    y = luminance / _PQ_PEAK_LUMINANCE
    # Raise to m1
    ym1 = np.power(y, _PQ_M1)
    # Numerator and denominator
    num = _PQ_C1 + _PQ_C2 * ym1
    den = 1.0 + _PQ_C3 * ym1
    # Final signal
    signal = np.power(num / den, _PQ_M2)
    return signal


# ============================================================
# HLG (ARIB STD-B67) — Hybrid Log-Gamma
# ============================================================
_HLG_A = 0.17883277
_HLG_B = 1.0 - 4.0 * _HLG_A  # 0.28466892
_HLG_C = 0.5 - _HLG_A * np.log(4.0 * _HLG_A)  # 0.55991073


def hlg_oetf(linear: NDArray[np.floating]) -> NDArray[np.floating]:
    """HLG OETF: scene-referred linear [0, 1] → HLG signal [0, 1]."""
    linear = np.clip(linear, 0.0, 1.0)
    result = np.where(
        linear <= 1.0 / 12.0,
        np.sqrt(3.0 * linear),
        _HLG_A * np.log(12.0 * linear - _HLG_B) + _HLG_C,
    )
    return result


def hlg_eotf_inv(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """HLG inverse OETF: HLG signal [0, 1] → scene-referred linear [0, 1]."""
    signal = np.clip(signal, 0.0, 1.0)
    result = np.where(
        signal <= 0.5,
        signal * signal / 3.0,
        (np.exp((signal - _HLG_C) / _HLG_A) + _HLG_B) / 12.0,
    )
    return result


# ============================================================
# BT.1886 — SDR Display EOTF (gamma ~2.4)
# ============================================================
_BT1886_GAMMA = 2.4


def bt1886_eotf(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """BT.1886 EOTF: signal [0, 1] → display linear luminance [0, 1].

    Simplified model assuming Lw=1, Lb=0.
    """
    signal = np.clip(signal, 0.0, 1.0)
    return np.power(signal, _BT1886_GAMMA)


def bt1886_oetf(linear: NDArray[np.floating]) -> NDArray[np.floating]:
    """BT.1886 inverse EOTF: display linear [0, 1] → signal [0, 1]."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.power(linear, 1.0 / _BT1886_GAMMA)


# ============================================================
# Utility: linearize / delinearize by transfer function name
# ============================================================


def linearize(
    signal: NDArray[np.floating], transfer: str, peak_nits: float = 10000.0
) -> NDArray[np.floating]:
    """Convert signal-domain values to linear luminance.

    Args:
        signal: Input values in [0, 1] signal domain.
        transfer: Transfer function identifier (e.g. "smpte2084", "bt709", "arib-std-b67").
        peak_nits: Peak luminance for normalization (used with PQ).

    Returns:
        Linear luminance values. For PQ, normalized to [0, 1] by dividing by peak_nits.
        For SDR and HLG, already in [0, 1].
    """
    transfer_lower = transfer.lower().replace("-", "").replace("_", "").replace(" ", "")

    if transfer_lower in ("smpte2084", "pq", "st2084"):
        return pq_eotf(signal) / peak_nits
    elif transfer_lower in ("aribstdb67", "hlg"):
        return hlg_eotf_inv(signal)
    elif transfer_lower in ("bt709", "bt1886", "gamma24", "iec6196621"):
        return bt1886_eotf(signal)
    else:
        # Fallback: assume gamma 2.2
        return np.power(np.clip(signal, 0.0, 1.0), 2.2)


def delinearize(
    linear: NDArray[np.floating], transfer: str, peak_nits: float = 10000.0
) -> NDArray[np.floating]:
    """Convert linear luminance to signal-domain values.

    Args:
        linear: Linear luminance values in [0, 1] (normalized).
        transfer: Target transfer function.
        peak_nits: Peak luminance for PQ.

    Returns:
        Signal-domain values in [0, 1].
    """
    transfer_lower = transfer.lower().replace("-", "").replace("_", "").replace(" ", "")

    if transfer_lower in ("smpte2084", "pq", "st2084"):
        return pq_oetf(linear * peak_nits)
    elif transfer_lower in ("aribstdb67", "hlg"):
        return hlg_oetf(np.clip(linear, 0.0, 1.0))
    elif transfer_lower in ("bt709", "bt1886", "gamma24", "iec6196621"):
        return bt1886_oetf(np.clip(linear, 0.0, 1.0))
    else:
        # Fallback: assume gamma 2.2
        return np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)
