"""Color metrics: deltaE2000, deltaE ICtCp, chroma error, hue error.

Implements perceptual color difference metrics for HDR quality evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import linear_bt2020_to_ictcp, linear_to_lab_d65


def delta_e_2000(
    predicted_lab: NDArray[np.floating], reference_lab: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Simplified CIEDE2000 color difference.

    Computes deltaE2000 between predicted and reference L*a*b* values.
    This is a simplified implementation suitable for research evaluation.

    Args:
        predicted_lab: Predicted L*a*b* values, shape (..., 3).
        reference_lab: Reference L*a*b* values, shape (..., 3).

    Returns:
        DeltaE2000 values, same shape as input minus last dimension.
    """
    L1, a1, b1 = predicted_lab[..., 0], predicted_lab[..., 1], predicted_lab[..., 2]
    L2, a2, b2 = reference_lab[..., 0], reference_lab[..., 1], reference_lab[..., 2]

    # Step 1: Calculate C'ab and h'ab
    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_avg = (C1 + C2) / 2.0

    C_avg_7 = np.power(C_avg, 7.0)
    G = 0.5 * (1.0 - np.sqrt(C_avg_7 / (C_avg_7 + 25.0**7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    # Step 2: Compute delta L', delta C', delta H'
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    # delta h'
    dh_prime = np.where(
        C1_prime * C2_prime == 0.0,
        0.0,
        np.where(
            np.abs(h2_prime - h1_prime) <= 180.0,
            h2_prime - h1_prime,
            np.where(
                h2_prime - h1_prime > 180.0,
                h2_prime - h1_prime - 360.0,
                h2_prime - h1_prime + 360.0,
            ),
        ),
    )

    dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2.0))

    # Step 3: Weighting functions
    L_avg = (L1 + L2) / 2.0
    C_avg_prime = (C1_prime + C2_prime) / 2.0

    # Average hue
    h_avg = np.where(
        C1_prime * C2_prime == 0.0,
        h1_prime + h2_prime,
        np.where(
            np.abs(h1_prime - h2_prime) <= 180.0,
            (h1_prime + h2_prime) / 2.0,
            np.where(
                h1_prime + h2_prime < 360.0,
                (h1_prime + h2_prime + 360.0) / 2.0,
                (h1_prime + h2_prime - 360.0) / 2.0,
            ),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_avg - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_avg))
        + 0.32 * np.cos(np.radians(3.0 * h_avg + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_avg - 63.0))
    )

    SL = 1.0 + 0.015 * (L_avg - 50.0) ** 2 / np.sqrt(20.0 + (L_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * C_avg_prime
    SH = 1.0 + 0.015 * C_avg_prime * T

    C_avg_prime_7 = np.power(C_avg_prime, 7.0)
    RT = (
        -2.0
        * np.sqrt(C_avg_prime_7 / (C_avg_prime_7 + 25.0**7))
        * np.sin(np.radians(60.0 * np.exp(-((h_avg - 275.0) / 25.0) ** 2)))
    )

    # Final deltaE2000
    term_L = dL_prime / SL
    term_C = dC_prime / SC
    term_H = dH_prime / SH

    dE = np.sqrt(term_L**2 + term_C**2 + term_H**2 + RT * term_C * term_H)
    return dE


def delta_e_ictcp(
    predicted_nits: NDArray[np.floating], reference_nits: NDArray[np.floating]
) -> NDArray[np.floating]:
    """DeltaE in ICtCp color space (perceptual HDR metric).

    Computes Euclidean distance in ICtCp space, which is perceptually
    uniform for HDR content.

    Args:
        predicted_nits: Predicted linear BT.2020 RGB in nits, shape (..., 3).
        reference_nits: Reference linear BT.2020 RGB in nits, shape (..., 3).

    Returns:
        DeltaE ICtCp values, same shape as input minus last dimension.
    """
    ictcp_pred = linear_bt2020_to_ictcp(predicted_nits)
    ictcp_ref = linear_bt2020_to_ictcp(reference_nits)

    diff = ictcp_pred - ictcp_ref
    # Weight: 1x I, 1x Ct, 1x Cp (standard ICtCp deltaE)
    dE = np.sqrt(np.sum(diff * diff, axis=-1))
    return dE


def chroma_error(
    predicted_nits: NDArray[np.floating], reference_nits: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Chroma magnitude error in ICtCp space.

    Computes the difference in chroma (saturation) magnitude between
    predicted and reference values.

    Args:
        predicted_nits: Predicted linear BT.2020 RGB in nits, shape (..., 3).
        reference_nits: Reference linear BT.2020 RGB in nits, shape (..., 3).

    Returns:
        Chroma magnitude error (positive = more saturated than reference).
    """
    ictcp_pred = linear_bt2020_to_ictcp(predicted_nits)
    ictcp_ref = linear_bt2020_to_ictcp(reference_nits)

    chroma_pred = np.sqrt(ictcp_pred[..., 1] ** 2 + ictcp_pred[..., 2] ** 2)
    chroma_ref = np.sqrt(ictcp_ref[..., 1] ** 2 + ictcp_ref[..., 2] ** 2)

    return chroma_pred - chroma_ref


def hue_error(
    predicted_nits: NDArray[np.floating], reference_nits: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Hue angle error in degrees (ICtCp space).

    Computes the angular difference in hue between predicted and reference.

    Args:
        predicted_nits: Predicted linear BT.2020 RGB in nits, shape (..., 3).
        reference_nits: Reference linear BT.2020 RGB in nits, shape (..., 3).

    Returns:
        Hue error in degrees, range [-180, 180].
    """
    ictcp_pred = linear_bt2020_to_ictcp(predicted_nits)
    ictcp_ref = linear_bt2020_to_ictcp(reference_nits)

    hue_pred = np.arctan2(ictcp_pred[..., 2], ictcp_pred[..., 1])
    hue_ref = np.arctan2(ictcp_ref[..., 2], ictcp_ref[..., 1])

    # Compute signed angular difference
    diff = hue_pred - hue_ref
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi

    return np.degrees(diff)
