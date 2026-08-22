"""Geometric alignment between HDR and Open Matte.

Determines the spatial relationship (scale + translation) between the HDR
master frame and the Open Matte frame using edge-based image alignment.

Algorithm:
1. Extract corresponding frame pairs (HDR frame N, OM frame N + sync_offset)
   at low-resolution proxy (~480px width).
2. Compute edge maps (Sobel) — transfer-function agnostic.
3. For each pair, search for the best vertical/horizontal translation
   that maximizes NCC between HDR edge map and a region of OM edge map.
4. Optionally search a small scale range around 1.0.
5. Repeat for multiple samples across the timeline.
6. Take robust median and verify stability.

Key assumptions:
- HDR content is a spatial subset of Open Matte (same scene, different crop).
- HDR is NEVER transformed — we find where HDR maps within OM.
- Both sources have the same pixel width (typical: 3840) but different heights.
- The dominant unknown is vertical offset (offset_y).
- Scale is expected to be ~1.0 but verified from the image.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import GeometryConfig
from auto_openmatte.core.exceptions import GeometryError
from auto_openmatte.core.models import GeometryModel, SourceInfo, SyncModel
from auto_openmatte.utils.frames import compute_edge_map, extract_segment_at_time_grayscale
from auto_openmatte.utils.math_utils import normalized_cross_correlation

logger = logging.getLogger(__name__)

# Number of sample frame pairs for geometry estimation
_N_SAMPLES = 7

# Proxy width for geometry analysis
_PROXY_WIDTH = 480

# Scale search range (fraction around 1.0)
_SCALE_SEARCH_RANGE = 0.03  # ±3%
_SCALE_SEARCH_STEPS = 7  # Number of scale candidates

# Minimum edge energy to consider a frame usable for geometry
_MIN_FRAME_ENERGY = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_pair_edges(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    hdr_frame: int,
    sync_offset: int,
    proxy_width: int = _PROXY_WIDTH,
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    """Extract edge maps for a corresponding HDR/OM frame pair.

    Returns (hdr_edge, om_edge) or (None, None) on failure.
    """
    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 24.0
    hdr_stream_idx = hdr_source.selected_stream.index if hdr_source.selected_stream else 0
    om_stream_idx = om_source.selected_stream.index if om_source.selected_stream else 0

    hdr_time = hdr_frame / hdr_fps
    om_frame = hdr_frame + sync_offset
    om_time = om_frame / hdr_fps

    # Extract single frame from each (use segment extraction with n_frames=1)
    hdr_frames = extract_segment_at_time_grayscale(
        hdr_source.path, hdr_time, n_frames=1,
        stream_index=hdr_stream_idx, width=proxy_width,
    )
    om_frames = extract_segment_at_time_grayscale(
        om_source.path, om_time, n_frames=1,
        stream_index=om_stream_idx, width=proxy_width,
    )

    if not hdr_frames or hdr_frames[0] is None:
        return None, None
    if not om_frames or om_frames[0] is None:
        return None, None

    hdr_gray = hdr_frames[0]
    om_gray = om_frames[0]

    # Check energy — skip very dark/blank frames
    if float(np.std(hdr_gray)) < _MIN_FRAME_ENERGY:
        return None, None
    if float(np.std(om_gray)) < _MIN_FRAME_ENERGY:
        return None, None

    hdr_edge = compute_edge_map(hdr_gray)
    om_edge = compute_edge_map(om_gray)

    return hdr_edge, om_edge


def _find_best_translation(
    hdr_edge: NDArray[np.floating],
    om_edge: NDArray[np.floating],
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """Find the best (offset_x, offset_y) that places HDR within OM.

    Slides the (possibly scaled) HDR edge map vertically and horizontally
    within the OM edge map, computing NCC at each position.

    Args:
        hdr_edge: HDR edge map (h_hdr, w_hdr).
        om_edge: OM edge map (h_om, w_om).
        scale: Scale factor to apply to HDR dimensions.

    Returns:
        (offset_x, offset_y, ncc_score) in proxy pixel coordinates.
        offset_x/y: position of HDR top-left corner within OM.
    """
    h_hdr, w_hdr = hdr_edge.shape
    h_om, w_om = om_edge.shape

    # Effective HDR size after scaling
    eff_h = int(round(h_hdr * scale))
    eff_w = int(round(w_hdr * scale))

    if eff_h > h_om or eff_w > w_om:
        # HDR larger than OM after scaling — can't fit
        return 0.0, 0.0, 0.0

    # If scale != 1, resize HDR edge map
    if abs(scale - 1.0) > 0.001:
        from scipy.ndimage import zoom
        hdr_scaled = zoom(hdr_edge, (scale, scale), order=1)
    else:
        hdr_scaled = hdr_edge
        eff_h, eff_w = h_hdr, w_hdr

    # Search range for vertical offset
    max_offset_y = h_om - eff_h
    max_offset_x = w_om - eff_w

    # Horizontal search: if same width, only ox=0; otherwise search full range
    # but limit to ±10 pixels from expected center to avoid excessive computation
    if max_offset_x <= 0:
        x_range = [0]
    elif max_offset_x <= 20:
        x_range = list(range(0, max_offset_x + 1))
    else:
        # Search around center ± 10
        x_center = max_offset_x // 2
        x_start = max(0, x_center - 10)
        x_end = min(max_offset_x, x_center + 11)
        x_range = list(range(x_start, x_end))

    best_score = -2.0
    best_ox = 0.0
    best_oy = 0.0

    for oy in range(0, max_offset_y + 1):
        for ox in x_range:
            # Extract OM region at this position
            om_region = om_edge[oy: oy + eff_h, ox: ox + eff_w]

            if om_region.shape != hdr_scaled.shape:
                continue

            score = normalized_cross_correlation(hdr_scaled, om_region)

            if score > best_score:
                best_score = score
                best_ox = float(ox)
                best_oy = float(oy)

    return best_ox, best_oy, best_score


def _estimate_single_pair(
    hdr_edge: NDArray[np.floating],
    om_edge: NDArray[np.floating],
) -> tuple[float, float, float, float, float]:
    """Estimate geometry from a single frame pair.

    Searches scale and translation to find best alignment.

    Returns:
        (scale_x, scale_y, offset_x, offset_y, confidence)
        All in proxy coordinates.
    """
    # Generate scale candidates centered on 1.0
    scales = np.linspace(
        1.0 - _SCALE_SEARCH_RANGE,
        1.0 + _SCALE_SEARCH_RANGE,
        _SCALE_SEARCH_STEPS,
    )

    best_scale = 1.0
    best_ox = 0.0
    best_oy = 0.0
    best_score = -2.0

    for scale in scales:
        ox, oy, score = _find_best_translation(hdr_edge, om_edge, scale=scale)
        if score > best_score:
            best_score = score
            best_scale = scale
            best_ox = ox
            best_oy = oy

    return best_scale, best_scale, best_ox, best_oy, best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_geometry(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    sync_model: SyncModel,
    config: GeometryConfig | None = None,
) -> GeometryModel:
    """Estimate geometric relationship between HDR and Open Matte.

    Uses edge-based spatial alignment on multiple frame pairs to determine
    scale, translation, and overlap region. HDR is never transformed.

    Args:
        hdr_source: Inspected HDR source.
        om_source: Inspected Open Matte source.
        sync_model: Validated sync model (provides frame_offset).
        config: Geometry configuration.

    Returns:
        GeometryModel with alignment parameters and overlap_bbox.

    Raises:
        GeometryError: If alignment cannot be determined with sufficient confidence.
    """
    if config is None:
        config = GeometryConfig()

    hdr_stream = hdr_source.selected_stream
    om_stream = om_source.selected_stream

    if not hdr_stream or not om_stream:
        raise GeometryError("Sources must have selected video streams.")

    hdr_w = hdr_stream.width
    hdr_h = hdr_stream.height
    om_w = om_stream.width
    om_h = om_stream.height
    hdr_fps = hdr_stream.fps if hdr_stream.fps > 0 else 24.0
    hdr_total = hdr_stream.frame_count or int(hdr_stream.duration_seconds * hdr_fps)
    sync_offset = sync_model.frame_offset

    logger.info(
        f"Geometry estimation: HDR {hdr_w}x{hdr_h}, OM {om_w}x{om_h}, "
        f"sync_offset={sync_offset}"
    )

    # Select sample frames distributed across timeline (avoiding first/last 10%)
    margin = max(100, int(hdr_total * 0.10))
    sample_positions = np.linspace(margin, hdr_total - margin, _N_SAMPLES, dtype=int).tolist()

    # Estimate geometry from each sample
    results: list[tuple[float, float, float, float, float]] = []

    for hdr_frame in sample_positions:
        hdr_edge, om_edge = _extract_pair_edges(
            hdr_source, om_source, hdr_frame, sync_offset, proxy_width=_PROXY_WIDTH
        )
        if hdr_edge is None or om_edge is None:
            logger.debug(f"  Frame {hdr_frame}: skipped (low-info or extraction failed)")
            continue

        sx, sy, ox, oy, score = _estimate_single_pair(hdr_edge, om_edge)
        results.append((sx, sy, ox, oy, score))
        logger.debug(
            f"  Frame {hdr_frame}: scale=({sx:.4f},{sy:.4f}), "
            f"offset=({ox:.1f},{oy:.1f}), score={score:.4f}"
        )

    if not results:
        raise GeometryError(
            "No valid frame pairs for geometry estimation. "
            "All samples were low-information or extraction failed."
        )

    # Robust consensus: median of results
    scales_x = np.array([r[0] for r in results])
    scales_y = np.array([r[1] for r in results])
    offsets_x = np.array([r[2] for r in results])
    offsets_y = np.array([r[3] for r in results])
    scores = np.array([r[4] for r in results])

    median_sx = float(np.median(scales_x))
    median_sy = float(np.median(scales_y))
    median_ox = float(np.median(offsets_x))
    median_oy = float(np.median(offsets_y))
    mean_score = float(np.mean(scores))

    # Convert from proxy coordinates to full resolution
    # proxy_width corresponds to full width; heights scale proportionally
    proxy_to_full_x = om_w / _PROXY_WIDTH
    # proxy height for OM: om_h * (_PROXY_WIDTH / om_w)
    proxy_to_full_y = om_w / _PROXY_WIDTH  # Same ratio since aspect preserved

    full_offset_x = median_ox * proxy_to_full_x
    full_offset_y = median_oy * proxy_to_full_y

    # Scale is dimensionless — same in proxy and full resolution
    final_sx = median_sx
    final_sy = median_sy

    # Check stability
    std_ox = float(np.std(offsets_x)) * proxy_to_full_x
    std_oy = float(np.std(offsets_y)) * proxy_to_full_y
    std_sx = float(np.std(scales_x))
    std_sy = float(np.std(scales_y))

    is_stable = (
        std_ox <= config.stability_tolerance
        and std_oy <= config.stability_tolerance
        and std_sx < 0.01
        and std_sy < 0.01
    )

    logger.info(
        f"Geometry results ({len(results)} samples): "
        f"scale=({final_sx:.4f}, {final_sy:.4f}), "
        f"offset=({full_offset_x:.1f}, {full_offset_y:.1f}), "
        f"std_offset=({std_ox:.1f}, {std_oy:.1f}), "
        f"mean_score={mean_score:.4f}, stable={is_stable}"
    )

    # Compute confidence
    # Based on: mean NCC score × stability factor
    stability_factor = 1.0
    if not is_stable:
        stability_factor = 0.7
    confidence = mean_score * stability_factor
    confidence = min(1.0, max(0.0, confidence))

    if confidence < config.min_confidence:
        logger.warning(
            f"Geometry confidence {confidence:.4f} below threshold "
            f"{config.min_confidence}. Alignment may be unreliable."
        )

    # Compute overlap_bbox in OM full-resolution coordinates
    # HDR region within OM: starts at (offset_x, offset_y), size = hdr_w*sx × hdr_h*sy
    bbox_x1 = full_offset_x
    bbox_y1 = full_offset_y
    bbox_x2 = full_offset_x + hdr_w * final_sx
    bbox_y2 = full_offset_y + hdr_h * final_sy

    # Clamp to OM dimensions
    bbox_x1 = max(0.0, bbox_x1)
    bbox_y1 = max(0.0, bbox_y1)
    bbox_x2 = min(float(om_w), bbox_x2)
    bbox_y2 = min(float(om_h), bbox_y2)

    geometry = GeometryModel(
        scale_x=final_sx,
        scale_y=final_sy,
        offset_x=full_offset_x,
        offset_y=full_offset_y,
        overlap_bbox=[bbox_x1, bbox_y1, bbox_x2, bbox_y2],
        confidence=confidence,
        is_global=is_stable,
        shot_id=None,
    )

    logger.info(
        f"Final geometry: scale=({geometry.scale_x:.4f}, {geometry.scale_y:.4f}), "
        f"offset=({geometry.offset_x:.1f}, {geometry.offset_y:.1f}), "
        f"overlap_bbox={[f'{v:.0f}' for v in geometry.overlap_bbox]}, "
        f"confidence={geometry.confidence:.4f}, is_global={geometry.is_global}"
    )

    return geometry
