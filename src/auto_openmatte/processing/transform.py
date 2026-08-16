"""Per-shot transform application.

Applies the complete SDR → HDR transformation chain to Open Matte pixels:
1. Linearize SDR (BT.1886 EOTF per-channel)
2. Convert BT.709 → BT.2020 gamut
3. Apply luminance curve (BT.2020 Y)
4. Apply color correction matrix
5. Apply saturation adjustment
6. Delinearize to HDR (PQ OETF)

Gamut conversion ensures the output has correct BT.2020 chromaticity,
consistent with how the luminance curve was fitted in sampling.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import delinearize, linearize
from auto_openmatte.processing.color import apply_color_matrix
from auto_openmatte.processing.luminance import apply_luminance_curve

# BT.709 → BT.2020 linear RGB conversion matrix (ITU-R BT.2087, D65 white)
# Same matrix as used in sampling.py for curve fitting consistency.
_M_709_TO_2020 = np.array([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690972, 0.9195404, 0.0113624],
    [0.0163914, 0.0880133, 0.8955953],
], dtype=np.float64)

# BT.2020 luminance weights (must match sampling.py)
_LUM_R_2020 = 0.2627
_LUM_G_2020 = 0.6780
_LUM_B_2020 = 0.0593


def apply_shot_transform(
    om_frame: NDArray[np.floating],
    transform: ShotTransform,
    sdr_transfer: str = "bt709",
    hdr_transfer: str = "smpte2084",
    peak_nits: float = 10000.0,
) -> NDArray[np.floating]:
    """Apply the complete SDR → HDR transform to an Open Matte frame.

    The transform is applied to the ENTIRE frame. The caller is responsible
    for compositing only the extension region with the HDR master.

    Pipeline:
    1. SDR signal [0,1] → per-channel BT.1886 EOTF → linear BT.709 RGB
    2. Linear BT.709 → M_709_TO_2020 → linear BT.2020 RGB
    3. Compute BT.2020 Y, apply luminance curve, ratio-scale RGB
    4. Optional: color correction matrix, saturation
    5. Linear BT.2020 → PQ OETF → HDR signal [0,1]

    Args:
        om_frame: Open Matte frame, shape (H, W, 3), values in [0, 1] signal domain.
        transform: Per-shot transform parameters.
        sdr_transfer: SDR transfer function identifier.
        hdr_transfer: HDR transfer function identifier.
        peak_nits: Peak luminance for PQ encoding.

    Returns:
        Transformed frame in HDR signal domain [0, 1], shape (H, W, 3).
    """
    # Step 1: Linearize SDR (per-channel EOTF)
    linear_709 = linearize(om_frame, sdr_transfer)

    # Step 2: Convert BT.709 → BT.2020
    shape = linear_709.shape
    linear_2020 = linear_709.reshape(-1, 3) @ _M_709_TO_2020.T
    linear_2020 = linear_2020.reshape(shape)
    linear_2020 = np.maximum(linear_2020, 0.0)

    # Step 3: Apply luminance curve in BT.2020 space
    linear = linear_2020
    if transform.luminance_curve and len(transform.luminance_curve) >= 2:
        # Compute BT.2020 luminance (consistent with sampling.py)
        lum = (
            _LUM_R_2020 * linear[..., 0]
            + _LUM_G_2020 * linear[..., 1]
            + _LUM_B_2020 * linear[..., 2]
        )

        # Apply curve to luminance
        lum_mapped = apply_luminance_curve(lum, transform.luminance_curve)

        # Scale RGB channels by luminance ratio
        safe_mask = lum > 1e-6
        ratio = np.ones_like(lum)
        np.divide(lum_mapped, lum, out=ratio, where=safe_mask)
        linear = linear * ratio[..., np.newaxis]
        linear = np.maximum(linear, 0.0)

    # Step 4: Apply color correction matrix (in BT.2020 space)
    if transform.color_matrix:
        mat = np.array(transform.color_matrix)
        if not np.allclose(mat, np.eye(3), atol=0.001):
            linear = apply_color_matrix(linear, transform.color_matrix)

    # Step 5: Apply saturation adjustment (in BT.2020 space)
    if abs(transform.saturation - 1.0) > 0.01:
        lum = (
            _LUM_R_2020 * linear[..., 0]
            + _LUM_G_2020 * linear[..., 1]
            + _LUM_B_2020 * linear[..., 2]
        )
        chroma = linear - lum[..., np.newaxis]
        linear = lum[..., np.newaxis] + chroma * transform.saturation
        linear = np.maximum(linear, 0.0)

    # Step 6: Delinearize to HDR signal domain (PQ OETF)
    hdr_signal = delinearize(linear, hdr_transfer, peak_nits=peak_nits)

    return np.clip(hdr_signal, 0.0, 1.0)
