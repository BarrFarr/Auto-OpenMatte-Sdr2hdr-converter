"""Per-shot transform application.

Applies the complete SDR → HDR transformation chain to Open Matte pixels:
1. Linearize SDR (BT.1886 EOTF)
2. Apply luminance curve
3. Apply color correction matrix
4. Apply saturation adjustment
5. Delinearize to HDR (PQ OETF)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import delinearize, linearize
from auto_openmatte.processing.color import apply_color_matrix
from auto_openmatte.processing.luminance import apply_luminance_curve


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

    Args:
        om_frame: Open Matte frame, shape (H, W, 3), values in [0, 1] signal domain.
        transform: Per-shot transform parameters.
        sdr_transfer: SDR transfer function identifier.
        hdr_transfer: HDR transfer function identifier.
        peak_nits: Peak luminance for PQ encoding.

    Returns:
        Transformed frame in HDR signal domain [0, 1], shape (H, W, 3).
    """
    # Step 1: Linearize SDR
    # Input is in SDR signal domain [0, 1]
    linear = linearize(om_frame, sdr_transfer)

    # Step 2: Apply luminance curve
    # Work on luminance channel (Y from RGB)
    # Simple approach: apply curve to each channel weighted by luminance ratio
    if transform.luminance_curve and len(transform.luminance_curve) >= 2:
        # Compute luminance (Rec.709 weights for SDR source)
        lum = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]

        # Apply curve to luminance
        lum_mapped = apply_luminance_curve(lum, transform.luminance_curve)

        # Scale RGB channels by luminance ratio
        ratio = np.where(lum > 1e-6, lum_mapped / lum, 1.0)
        linear = linear * ratio[..., np.newaxis]
        linear = np.maximum(linear, 0.0)

    # Step 3: Apply color correction matrix
    if transform.color_matrix:
        # Check if matrix is identity (skip if so)
        mat = np.array(transform.color_matrix)
        if not np.allclose(mat, np.eye(3), atol=0.001):
            linear = apply_color_matrix(linear, transform.color_matrix)

    # Step 4: Apply saturation adjustment
    if abs(transform.saturation - 1.0) > 0.01:
        lum = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        chroma = linear - lum[..., np.newaxis]
        linear = lum[..., np.newaxis] + chroma * transform.saturation
        linear = np.maximum(linear, 0.0)

    # Step 5: Delinearize to HDR signal domain
    # Normalize linear values for PQ encoding
    # The luminance curve should already map to the correct HDR range
    hdr_signal = delinearize(linear, hdr_transfer, peak_nits=peak_nits)

    return np.clip(hdr_signal, 0.0, 1.0)
