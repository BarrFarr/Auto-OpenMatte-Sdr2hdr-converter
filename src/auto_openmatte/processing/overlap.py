"""Overlap region extraction and mask generation.

Identifies the common region between HDR and Open Matte,
and generates the extension mask for composition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.models import GeometryModel, SourceInfo


def generate_extension_mask(
    om_source: SourceInfo,
    geometry: GeometryModel,
    feather_width: int = 4,
) -> NDArray[np.floating]:
    """Generate the extension mask for composition.

    The mask defines which pixels come from Open Matte (1.0)
    vs which come from HDR (0.0), with a feathered transition.

    The mask is in Open Matte coordinate space (full OM resolution).

    Args:
        om_source: Open Matte source info.
        geometry: Geometry model defining overlap.
        feather_width: Width of feather/blend at HDR boundary.

    Returns:
        Float mask array (om_height, om_width) where:
        0.0 = HDR region (use original HDR pixels)
        1.0 = Extension region (use transformed OM pixels)
        0.0-1.0 = Feather/blend region
    """
    om_stream = om_source.selected_stream
    if not om_stream:
        return np.ones((1080, 1920), dtype=np.float64)

    om_h = om_stream.height
    om_w = om_stream.width

    # Start with all-extension (1.0)
    mask = np.ones((om_h, om_w), dtype=np.float64)

    # The overlap bbox defines where HDR content exists in OM coordinates
    x1, y1, x2, y2 = geometry.overlap_bbox

    # Clamp to valid range
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(om_w, int(round(x2)))
    y2 = min(om_h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        # No valid overlap — everything is extension
        return mask

    # Set the HDR region to 0.0 (will use HDR pixels here)
    # But with feather at the boundary
    inner_y1 = y1 + feather_width
    inner_y2 = y2 - feather_width
    inner_x1 = x1 + feather_width
    inner_x2 = x2 - feather_width

    # Core HDR region (no blending)
    if inner_y1 < inner_y2 and inner_x1 < inner_x2:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0.0

    # Feather regions (top edge of HDR)
    if feather_width > 0 and y1 < inner_y1:
        for row in range(y1, min(inner_y1, om_h)):
            alpha = (row - y1) / feather_width
            alpha = 1.0 - alpha  # 1.0 at y1, 0.0 at inner_y1
            mask[row, x1:x2] = alpha

    # Feather regions (bottom edge of HDR)
    if feather_width > 0 and inner_y2 < y2:
        for row in range(max(inner_y2, 0), y2):
            alpha = (row - inner_y2) / feather_width
            mask[row, x1:x2] = alpha  # 0.0 at inner_y2, 1.0 at y2

    # Feather regions (left edge of HDR) — usually not needed if full width
    if feather_width > 0 and x1 > 0 and x1 < inner_x1:
        for col in range(x1, min(inner_x1, om_w)):
            alpha = (col - x1) / feather_width
            alpha = 1.0 - alpha
            mask[y1:y2, col] = np.minimum(mask[y1:y2, col], alpha)

    # Feather regions (right edge of HDR) — usually not needed if full width
    if feather_width > 0 and inner_x2 < x2 and x2 < om_w:
        for col in range(max(inner_x2, 0), x2):
            alpha = (col - inner_x2) / feather_width
            mask[y1:y2, col] = np.minimum(mask[y1:y2, col], alpha)

    return mask


def extract_overlap_region(
    hdr_frame: NDArray[np.floating],
    om_frame: NDArray[np.floating],
    geometry: GeometryModel,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract the overlapping region from both frames.

    Returns corresponding pixel pairs from the region where HDR and OM
    show the same content, for use in luminance/color estimation.

    Args:
        hdr_frame: HDR frame (hdr_h, hdr_w, channels) in linear float.
        om_frame: OM frame (om_h, om_w, channels) in linear float.
        geometry: Geometry model.

    Returns:
        Tuple of (hdr_pixels, om_pixels) both shape (N, channels).
    """
    x1, y1, x2, y2 = geometry.overlap_bbox

    # Clamp to OM dimensions
    om_h, om_w = om_frame.shape[:2]
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(om_w, int(round(x2)))
    y2 = min(om_h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        # Return empty arrays
        channels = hdr_frame.shape[2] if hdr_frame.ndim == 3 else 1
        return np.empty((0, channels)), np.empty((0, channels))

    # Extract OM overlap region
    om_region = om_frame[y1:y2, x1:x2]

    # The HDR frame corresponds to this region after scaling
    # HDR covers the full frame, so we can use it directly
    # (the scale_x/scale_y maps HDR pixels to OM pixels)
    hdr_h, hdr_w = hdr_frame.shape[:2]

    # Resize HDR to match the overlap region size for pixel correspondence
    from scipy.ndimage import zoom

    region_h = y2 - y1
    region_w = x2 - x1

    if hdr_frame.ndim == 3:
        zoom_factors = (region_h / hdr_h, region_w / hdr_w, 1.0)
    else:
        zoom_factors = (region_h / hdr_h, region_w / hdr_w)

    hdr_resized = zoom(hdr_frame, zoom_factors, order=1)

    # Ensure same shape
    min_h = min(hdr_resized.shape[0], om_region.shape[0])
    min_w = min(hdr_resized.shape[1], om_region.shape[1])
    hdr_crop = hdr_resized[:min_h, :min_w]
    om_crop = om_region[:min_h, :min_w]

    # Flatten to (N, channels)
    if hdr_crop.ndim == 2:
        hdr_pixels = hdr_crop.reshape(-1, 1)
        om_pixels = om_crop.reshape(-1, 1)
    else:
        hdr_pixels = hdr_crop.reshape(-1, hdr_crop.shape[2])
        om_pixels = om_crop.reshape(-1, om_crop.shape[2])

    return hdr_pixels, om_pixels
