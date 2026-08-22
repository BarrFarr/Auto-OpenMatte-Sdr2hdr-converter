"""SDR simulator: controlled HDR-to-SDR conversion for ground truth generation.

Applies known tone mapping, gamut compression, and gamma encoding to
create SDR from HDR with a precisely known transformation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils.color_spaces import bt2020_to_bt709, gamut_clip_bt709, BT2020_LUMA
from ..utils.transfer_functions import bt1886_oetf


def simulate_sdr(
    hdr_linear_bt2020: NDArray[np.floating],
    tone_map_knee: float = 1000.0,
    tone_map_max: float = 4000.0,
    gamma: float = 2.4,
) -> NDArray[np.floating]:
    """Convert linear BT.2020 HDR (nits) to gamma-encoded BT.709 SDR.

    Applies a controlled pipeline:
    1. Reinhard-like tone mapping (compress HDR range to [0, 1])
    2. BT.2020 -> BT.709 gamut conversion with clipping
    3. Gamma encoding (BT.1886 inverse)

    Args:
        hdr_linear_bt2020: Linear BT.2020 RGB in nits, shape (..., 3).
        tone_map_knee: Luminance (nits) where tone curve starts rolling off.
        tone_map_max: Maximum HDR luminance to map (nits above this clip).
        gamma: Display gamma for encoding (default 2.4 for BT.1886).

    Returns:
        SDR image in gamma-encoded BT.709, values in [0, 1], shape (..., 3).
    """
    # Step 1: Tone mapping (modified Reinhard per-channel, luminance-guided)
    # Compute luminance for adaptation
    lum = np.einsum("...c,c->...", hdr_linear_bt2020, BT2020_LUMA)
    lum = np.maximum(lum, 1e-6)

    # Tone map operator: S-curve based on luminance
    # Maps [0, tone_map_max] -> [0, 1] with knee at tone_map_knee
    tone_mapped_lum = _reinhard_extended(lum, tone_map_knee, tone_map_max)

    # Apply luminance ratio to RGB (preserves color ratios)
    ratio = tone_mapped_lum / lum
    tone_mapped_rgb = hdr_linear_bt2020 * ratio[..., np.newaxis]

    # Step 2: Gamut conversion BT.2020 -> BT.709
    rgb709 = bt2020_to_bt709(tone_mapped_rgb)

    # Step 3: Gamut clip (simple clip to [0, 1])
    rgb709_clipped = gamut_clip_bt709(rgb709)

    # Step 4: Gamma encode (inverse EOTF)
    sdr_signal = bt1886_oetf(rgb709_clipped)

    return sdr_signal


def _reinhard_extended(
    luminance: NDArray[np.floating],
    knee: float,
    max_lum: float,
) -> NDArray[np.floating]:
    """Extended Reinhard tone mapping operator.

    Provides a soft roll-off starting at the knee point, mapping
    [0, max_lum] to approximately [0, 1].

    Args:
        luminance: Input luminance in nits.
        knee: Knee point where roll-off begins (nits).
        max_lum: Maximum input luminance (nits).

    Returns:
        Tone-mapped luminance in [0, 1].
    """
    # Normalize to [0, 1] range relative to max
    L_norm = luminance / max_lum

    # Extended Reinhard: L / (1 + L) with white point
    L_white = 1.0  # Normalized white point
    numerator = L_norm * (1.0 + L_norm / (L_white * L_white))
    denominator = 1.0 + L_norm

    result = numerator / denominator

    # Ensure output is in [0, 1]
    return np.clip(result, 0.0, 1.0)
