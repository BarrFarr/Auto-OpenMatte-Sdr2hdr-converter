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
    Only extension regions of the Open Matte frame are transformed (not the
    full frame), saving ~74% computation for typical 21:9→16:9 geometry.

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

    # Determine overlap region
    x1, y1, x2, y2 = geometry.overlap_bbox
    x1, y1 = int(round(x1)), int(round(y1))
    x2, y2 = int(round(x2)), int(round(y2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(om_w, x2)
    y2 = min(om_h, y2)

    # Initialize output canvas
    output = np.zeros((om_h, om_w, 3), dtype=om_frame.dtype)

    # Step 1: Transform ONLY extension regions (top + bottom)
    # Top extension: rows 0 to y1
    if y1 > 0:
        top_ext = om_frame[:y1, :, :]
        top_transformed = apply_shot_transform(
            top_ext, transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        output[:y1, :, :] = top_transformed

    # Bottom extension: rows y2 to om_h
    if y2 < om_h:
        bot_ext = om_frame[y2:, :, :]
        bot_transformed = apply_shot_transform(
            bot_ext, transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        output[y2:, :, :] = bot_transformed

    # Step 2: Place HDR in overlap region
    region_h = y2 - y1
    region_w = x2 - x1

    if region_h > 0 and region_w > 0:
        hdr_h, hdr_w = hdr_frame.shape[:2]

        # Skip resize if scale is identity (common case: same width, matching height)
        if hdr_h == region_h and hdr_w == region_w:
            hdr_placed = hdr_frame
        else:
            from scipy.ndimage import zoom
            zoom_factors = (region_h / hdr_h, region_w / hdr_w, 1.0)
            hdr_placed = zoom(hdr_frame, zoom_factors, order=1)
            hdr_placed = hdr_placed[:region_h, :region_w, :]

        # Step 3: Blend in overlap region using mask
        # For the feather zone: transform OM overlap pixels too
        mask_region = extension_mask[y1:y2, x1:x2]
        has_feather = np.any((mask_region > 0.0) & (mask_region < 1.0))

        if has_feather:
            # Need transformed OM in the overlap for feather blending
            overlap_om = om_frame[y1:y2, x1:x2, :]
            overlap_transformed = apply_shot_transform(
                overlap_om, transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
            # Blend: (1-mask)*HDR + mask*OM_transformed
            for ch in range(3):
                output[y1:y2, x1:x2, ch] = (
                    (1.0 - mask_region) * hdr_placed[:, :, ch]
                    + mask_region * overlap_transformed[:, :, ch]
                )
        else:
            # Pure HDR placement (mask is 0 everywhere in overlap)
            output[y1:y2, x1:x2, :] = hdr_placed

    # Left/right extension (if overlap doesn't span full width)
    if x1 > 0 or x2 < om_w:
        # Transform left/right strips if they exist
        if x1 > 0:
            left = om_frame[y1:y2, :x1, :]
            output[y1:y2, :x1, :] = apply_shot_transform(
                left, transform, sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer, peak_nits=peak_nits,
            )
        if x2 < om_w:
            right = om_frame[y1:y2, x2:, :]
            output[y1:y2, x2:, :] = apply_shot_transform(
                right, transform, sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer, peak_nits=peak_nits,
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
