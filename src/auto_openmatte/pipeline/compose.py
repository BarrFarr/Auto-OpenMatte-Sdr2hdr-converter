"""Frame composition — combine HDR master with transformed Open Matte.

Mode A (extend): HDR center + transformed OM extension areas
Mode B (convert-hdr): Entire OM frame transformed to HDR
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.models import GeometryModel, ShotTransform
from auto_openmatte.processing.transform import apply_shot_transform


def composite_extend(
    hdr_frame: NDArray[np.floating],
    om_frame: NDArray[np.floating],
    transform: ShotTransform,
    geometry: GeometryModel,
    extension_mask: NDArray[np.floating],
    sdr_transfer: str = "bt709",
    hdr_transfer: str = "smpte2084",
    peak_nits: float = 10000.0,
) -> NDArray[np.floating]:
    """Composite Mode A: HDR center + transformed OM extension.

    The HDR frame is placed in its original position (MASTER).
    The Open Matte frame is transformed and used only in extension areas.

    Args:
        hdr_frame: HDR frame (hdr_h, hdr_w, 3) in signal domain [0, 1].
        om_frame: Open Matte frame (om_h, om_w, 3) in signal domain [0, 1].
        transform: Per-shot transform to apply to OM.
        geometry: Geometry model defining overlap.
        extension_mask: Float mask (om_h, om_w) — 0=HDR, 1=OM extension.
        sdr_transfer: SDR transfer function for OM.
        hdr_transfer: HDR transfer function for output.
        peak_nits: Peak luminance.

    Returns:
        Composited frame (om_h, om_w, 3) in HDR signal domain [0, 1].
    """
    om_h, om_w = om_frame.shape[:2]

    # Step 1: Transform entire OM frame to HDR
    om_transformed = apply_shot_transform(
        om_frame, transform,
        sdr_transfer=sdr_transfer,
        hdr_transfer=hdr_transfer,
        peak_nits=peak_nits,
    )

    # Step 2: Create output canvas (start with transformed OM)
    output = om_transformed.copy()

    # Step 3: Place HDR in its region
    # The HDR frame needs to be scaled to fit the overlap region
    x1, y1, x2, y2 = geometry.overlap_bbox
    x1, y1 = int(round(x1)), int(round(y1))
    x2, y2 = int(round(x2)), int(round(y2))

    # Clamp to valid range
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(om_w, x2)
    y2 = min(om_h, y2)

    region_h = y2 - y1
    region_w = x2 - x1

    if region_h > 0 and region_w > 0:
        # Resize HDR to fit the overlap region
        from scipy.ndimage import zoom

        hdr_h, hdr_w = hdr_frame.shape[:2]
        zoom_factors = (region_h / hdr_h, region_w / hdr_w, 1.0)
        hdr_resized = zoom(hdr_frame, zoom_factors, order=1)

        # Ensure exact size match
        hdr_resized = hdr_resized[:region_h, :region_w, :]

        # Step 4: Blend using extension mask
        # Where mask=0 → HDR pixels, where mask=1 → OM transformed pixels
        mask_region = extension_mask[y1:y2, x1:x2]

        # Apply blending in the overlap region
        for ch in range(3):
            output[y1:y2, x1:x2, ch] = (
                (1.0 - mask_region) * hdr_resized[:, :, ch]
                + mask_region * om_transformed[y1:y2, x1:x2, ch]
            )

    return output


def composite_convert_hdr(
    om_frame: NDArray[np.floating],
    transform: ShotTransform,
    sdr_transfer: str = "bt709",
    hdr_transfer: str = "smpte2084",
    peak_nits: float = 10000.0,
) -> NDArray[np.floating]:
    """Mode B: Transform entire OM frame to HDR (standalone conversion).

    No HDR center region — the entire frame is transformed.

    Args:
        om_frame: Open Matte frame (H, W, 3) in signal domain [0, 1].
        transform: Per-shot transform.
        sdr_transfer: SDR transfer function.
        hdr_transfer: HDR transfer function.
        peak_nits: Peak luminance.

    Returns:
        HDR frame (H, W, 3) in HDR signal domain [0, 1].
    """
    return apply_shot_transform(
        om_frame, transform,
        sdr_transfer=sdr_transfer,
        hdr_transfer=hdr_transfer,
        peak_nits=peak_nits,
    )
