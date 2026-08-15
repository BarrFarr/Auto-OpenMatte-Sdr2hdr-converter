"""Temporal synchronization — find global frame offset between HDR and Open Matte.

Algorithm overview:
1. Extract grayscale frames at low resolution (~480px width) using batch
   FFmpeg pipe (single subprocess per segment, NOT per frame).
2. Compute edge-magnitude maps (Sobel) — transfer-function agnostic.
3. Build temporal fingerprints: frame-to-frame temporal difference magnitude.
4. Use cross-correlation of temporal-difference sequences to find the integer
   frame offset that best aligns the two timelines.
5. Multi-window sampling: extract fingerprints from multiple independent
   positions across the film, compute offset per window, take robust consensus.
6. Validate by checking offset consistency at multiple checkpoints across
   the entire duration.

Key design decisions:
- Edge maps, NOT raw pixel luminance, are the primary feature — this avoids
  false mismatches between PQ and gamma/BT.1886 encoded sources.
- Temporal differences (frame-to-frame delta energy) form the matching signal,
  which is inherently content-driven and insensitive to static DC offset.
- Low-information segments (black, static, fades) are down-weighted via
  an energy threshold so they don't corrupt the offset estimate.
- Multi-window consensus prevents reliance on a single potentially bad segment.
- Batch FFmpeg extraction: one subprocess per segment (~5 segments for global
  offset), not per frame. Reduces subprocess count from ~9000 to ~15.
- phaseCorrelate is NOT used for temporal offset.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import SyncConfig
from auto_openmatte.core.exceptions import SyncDriftError, SynchronizationError
from auto_openmatte.core.models import (
    FrameRateType,
    SourceInfo,
    SyncModel,
    SyncStatus,
)
from auto_openmatte.utils.frames import (
    compute_edge_map,
    extract_segment_at_time_grayscale,
)
from auto_openmatte.utils.math_utils import normalized_cross_correlation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _center_crop(
    arr: NDArray[np.floating], target_h: int, target_w: int
) -> NDArray[np.floating]:
    """Crop a 2-D array to (target_h, target_w) from the center.

    If the array is already the target size, it is returned unchanged.
    The crop is symmetric — equal margins are removed from both sides.

    Args:
        arr: 2-D numpy array (H, W).
        target_h: Desired height.
        target_w: Desired width.

    Returns:
        Cropped array of shape (target_h, target_w).
    """
    h, w = arr.shape[:2]
    # Clamp target to actual size
    target_h = min(target_h, h)
    target_w = min(target_w, w)

    y_start = (h - target_h) // 2
    x_start = (w - target_w) // 2
    return arr[y_start: y_start + target_h, x_start: x_start + target_w]


def _get_total_frames(source: SourceInfo) -> int:
    """Determine total frame count for a source.

    Prefers the explicit frame_count from stream metadata.
    Falls back to duration * fps if frame_count is unavailable.

    Args:
        source: Inspected source information with selected_stream.

    Returns:
        Integer total frame count.
    """
    stream = source.selected_stream
    if stream is None:
        return 0
    if stream.frame_count is not None and stream.frame_count > 0:
        return stream.frame_count
    if stream.duration_seconds > 0 and stream.fps > 0:
        return int(round(stream.duration_seconds * stream.fps))
    return 0


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

# Minimum temporal-difference energy to consider a transition "real".
_MIN_DIFF_ENERGY = 0.005

# Number of sampling windows for multi-window global offset
_DEFAULT_N_WINDOWS = 5

# Seconds per sampling window
_WINDOW_SECONDS = 15.0

# Minimum fraction of windows that must agree for consensus
_MIN_CONSENSUS_RATIO = 0.5


def _compute_frame_energy(edge_map: NDArray[np.floating]) -> float:
    """Mean energy of an edge map — proxy for information content."""
    return float(np.mean(edge_map))


def _compute_temporal_diff(
    edge_a: NDArray[np.floating], edge_b: NDArray[np.floating]
) -> float:
    """Mean absolute difference between two edge maps.

    Both maps must have the same shape. If they differ (e.g. slight
    resolution mismatch between sources), center-crop to the smaller.
    """
    min_h = min(edge_a.shape[0], edge_b.shape[0])
    min_w = min(edge_a.shape[1], edge_b.shape[1])
    a = _center_crop(edge_a, min_h, min_w)
    b = _center_crop(edge_b, min_h, min_w)
    return float(np.mean(np.abs(a - b)))


def _build_diff_signal_batch(
    source: SourceInfo,
    start_seconds: float,
    n_frames: int,
    proxy_width: int = 480,
) -> NDArray[np.floating]:
    """Build temporal-difference signal using batch frame extraction.

    Extracts n_frames starting at start_seconds using a SINGLE FFmpeg process,
    computes edge maps, then frame-to-frame differences.

    Returns:
        1-D array of temporal differences (length n_frames - 1).
    """
    stream = source.selected_stream
    stream_index = stream.index if stream else 0

    frames = extract_segment_at_time_grayscale(
        source.path,
        start_seconds,
        n_frames,
        stream_index=stream_index,
        width=proxy_width,
    )

    # Compute edge maps and temporal differences
    diffs = np.zeros(max(0, n_frames - 1), dtype=np.float64)
    prev_edge: NDArray[np.floating] | None = None

    for i, frame in enumerate(frames):
        if frame is None:
            prev_edge = None
            continue

        edge = compute_edge_map(frame)

        if prev_edge is not None and i > 0:
            diffs[i - 1] = _compute_temporal_diff(edge, prev_edge)

        prev_edge = edge

    return diffs


# ---------------------------------------------------------------------------
# Cross-correlation based offset detection
# ---------------------------------------------------------------------------


def _find_best_offset_ncc(
    hdr_signal: NDArray[np.floating],
    om_signal: NDArray[np.floating],
    max_offset: int,
) -> tuple[int, float, float]:
    """Find the integer offset that maximizes NCC between two 1-D signals.

    Slides om_signal over hdr_signal in the range [-max_offset, +max_offset].
    The offset convention: om_frame = hdr_frame + offset.

    Returns:
        (best_offset, best_score, peak_ratio) where peak_ratio is
        best_score / second_best_score (higher = more unambiguous).
    """
    n_hdr = len(hdr_signal)
    n_om = len(om_signal)

    best_offset = 0
    best_score = -2.0
    second_best_score = -2.0

    # Minimum overlap length for a meaningful correlation
    min_overlap = max(20, min(n_hdr, n_om) // 4)

    for offset in range(-max_offset, max_offset + 1):
        # Determine overlapping region
        hdr_start = max(0, -offset)
        hdr_end = min(n_hdr, n_om - offset)
        om_start = max(0, offset)
        om_end = min(n_om, n_hdr + offset)

        overlap_len = min(hdr_end - hdr_start, om_end - om_start)
        if overlap_len < min_overlap:
            continue

        hdr_slice = hdr_signal[hdr_start: hdr_start + overlap_len]
        om_slice = om_signal[om_start: om_start + overlap_len]

        # Skip if both signals are essentially flat (no information)
        if np.std(hdr_slice) < _MIN_DIFF_ENERGY or np.std(om_slice) < _MIN_DIFF_ENERGY:
            continue

        score = normalized_cross_correlation(
            hdr_slice.reshape(1, -1), om_slice.reshape(1, -1)
        )

        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_offset = offset
        elif score > second_best_score:
            second_best_score = score

    # Peak ratio: how much better is the best vs second best
    if second_best_score > 0:
        peak_ratio = best_score / second_best_score
    elif best_score > 0:
        peak_ratio = 10.0  # No meaningful second peak
    else:
        peak_ratio = 1.0  # Both bad

    return best_offset, best_score, peak_ratio


# ---------------------------------------------------------------------------
# Multi-window offset estimation
# ---------------------------------------------------------------------------


def _estimate_offset_single_window(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    hdr_start_seconds: float,
    window_frames: int,
    max_offset_frames: int,
    proxy_width: int = 480,
) -> tuple[int, float, float]:
    """Estimate offset from a single temporal window.

    Args:
        hdr_source: HDR source.
        om_source: Open Matte source.
        hdr_start_seconds: Start time for HDR window.
        window_frames: Number of frames in the window.
        max_offset_frames: Maximum offset to search.
        proxy_width: Proxy width for extraction.

    Returns:
        (offset, ncc_score, peak_ratio) or (0, -1.0, 1.0) on failure.
    """
    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 24.0

    # OM window: wider to accommodate offset search
    # OM starts earlier to cover negative offsets, ends later for positive
    om_start_seconds = max(0.0, hdr_start_seconds - max_offset_frames / hdr_fps)
    om_n_frames = window_frames + 2 * max_offset_frames

    # Extract HDR diff signal (batch — single FFmpeg process)
    hdr_signal = _build_diff_signal_batch(
        hdr_source, hdr_start_seconds, window_frames, proxy_width=proxy_width
    )

    # Extract OM diff signal (batch — single FFmpeg process)
    om_signal = _build_diff_signal_batch(
        om_source, om_start_seconds, om_n_frames, proxy_width=proxy_width
    )

    # Check signal quality
    if np.std(hdr_signal) < _MIN_DIFF_ENERGY or np.std(om_signal) < _MIN_DIFF_ENERGY:
        return 0, -1.0, 1.0  # Low-information window

    # Search range in signal space
    search_range = max_offset_frames + int(
        abs(hdr_start_seconds - om_start_seconds) * hdr_fps
    )

    best_offset_signal, score, peak_ratio = _find_best_offset_ncc(
        hdr_signal, om_signal, max_offset=search_range
    )

    # Convert signal-space offset to absolute frame offset
    # hdr_signal corresponds to frames starting at hdr_start_seconds * fps
    # om_signal corresponds to frames starting at om_start_seconds * fps
    hdr_start_frame = int(round(hdr_start_seconds * hdr_fps))
    om_start_frame = int(round(om_start_seconds * hdr_fps))
    frame_offset = om_start_frame + best_offset_signal - hdr_start_frame

    return frame_offset, score, peak_ratio


def _multi_window_consensus(
    results: list[tuple[int, float, float]],
    min_confidence: float,
) -> tuple[int, float]:
    """Determine global offset from multiple window results via robust consensus.

    Groups offsets that differ by at most ±1 frame (to handle FFmpeg seeking
    imprecision), then selects the largest group. Within a group, the offset
    with the highest weighted score is selected as representative.

    Confidence is based on the fraction of agreeing windows and their scores.

    Args:
        results: List of (offset, score, peak_ratio) per window.
        min_confidence: Minimum per-window score to consider valid.

    Returns:
        (consensus_offset, consensus_confidence)
    """
    # Filter: only windows with positive, meaningful scores
    valid = [(off, sc, pr) for off, sc, pr in results if sc > 0.3]

    if not valid:
        return 0, 0.0

    # Group offsets with ±1 frame tolerance.
    # Strategy: sort by offset, then assign each to the group whose
    # representative is within ±1. Use the highest-scoring offset
    # as the group representative.
    # This prevents chain-grouping: 100→101→102 because each new
    # candidate is compared to the REPRESENTATIVE, not to all members.
    groups: list[tuple[int, list[tuple[int, float, float]]]] = []

    # Sort valid results by score descending — best scores establish groups first
    sorted_valid = sorted(valid, key=lambda x: -x[1])

    for off, sc, pr in sorted_valid:
        placed = False
        for i, (rep, members) in enumerate(groups):
            if abs(off - rep) <= 1:
                members.append((off, sc, pr))
                placed = True
                break
        if not placed:
            # New group — this offset becomes representative
            groups.append((off, [(off, sc, pr)]))

    # Select the largest group (by member count), breaking ties by mean score
    best_group_rep, best_group_members = max(
        groups,
        key=lambda g: (len(g[1]), np.mean([sc for _, sc, _ in g[1]])),
    )

    # Within the best group, select the offset with highest total weighted score
    # (score × peak_ratio). This picks the most confident individual measurement.
    offset_scores: dict[int, float] = {}
    for off, sc, pr in best_group_members:
        if off not in offset_scores:
            offset_scores[off] = 0.0
        offset_scores[off] += sc

    best_offset = max(offset_scores.keys(), key=lambda o: offset_scores[o])

    n_agreeing = len(best_group_members)
    n_valid = len(valid)
    n_total = len(results)

    # Consensus confidence:
    # - Based on agreement ratio among valid windows
    # - Weighted by mean NCC score of agreeing windows
    agreement_ratio = n_agreeing / n_valid if n_valid > 0 else 0.0
    mean_score = float(np.mean([sc for _, sc, _ in best_group_members]))

    # Combined confidence: agreement × mean_score
    confidence = agreement_ratio * mean_score

    # Penalty if too few windows are valid
    if n_valid < n_total * 0.4:
        confidence *= 0.7  # Significant portion of film was uninformative

    # Penalty for peak_ratio ambiguity in agreeing windows
    mean_peak_ratio = float(np.mean([pr for _, _, pr in best_group_members]))
    if mean_peak_ratio < 1.05:
        confidence *= 0.85  # Ambiguous peaks

    return best_offset, min(1.0, confidence)


# ---------------------------------------------------------------------------
# Public API: find_global_offset
# ---------------------------------------------------------------------------


def find_global_offset(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    config: SyncConfig | None = None,
) -> SyncModel:
    """Find the global frame offset between HDR and Open Matte.

    The offset convention is: OM_frame = HDR_frame + frame_offset.
    Positive offset means Open Matte starts later than HDR.

    Algorithm:
    1. Select multiple sampling windows distributed across the HDR timeline.
    2. For each window, extract temporal-difference signals via batch FFmpeg.
    3. Cross-correlate each window to find per-window offset candidates.
    4. Take robust consensus across all windows.
    5. Report confidence based on agreement and NCC quality.

    Args:
        hdr_source: Inspected HDR source with selected_stream set.
        om_source: Inspected Open Matte source with selected_stream set.
        config: Synchronization configuration.

    Returns:
        SyncModel with frame_offset, confidence, and status.

    Raises:
        SynchronizationError: If synchronization cannot be determined.
    """
    if config is None:
        config = SyncConfig()

    hdr_total = _get_total_frames(hdr_source)
    om_total = _get_total_frames(om_source)

    if hdr_total <= 0 or om_total <= 0:
        raise SynchronizationError(
            "Cannot determine frame counts for synchronization. "
            f"HDR frames={hdr_total}, OM frames={om_total}."
        )

    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 0.0
    om_fps = om_source.selected_stream.fps if om_source.selected_stream else 0.0

    if hdr_fps <= 0 or om_fps <= 0:
        raise SynchronizationError(
            f"Invalid FPS for synchronization: HDR={hdr_fps}, OM={om_fps}."
        )

    # Check for VFR — warn but continue with best-effort
    hdr_vfr = (
        hdr_source.selected_stream is not None
        and hdr_source.selected_stream.frame_rate_type == FrameRateType.VFR
    )
    om_vfr = (
        om_source.selected_stream is not None
        and om_source.selected_stream.frame_rate_type == FrameRateType.VFR
    )

    if hdr_vfr or om_vfr:
        logger.warning(
            "VFR detected — frame-locked synchronization may be unreliable. "
            "Proceeding with best-effort frame-index matching."
        )

    # Maximum offset in frames
    max_offset_frames = int(config.search_range_seconds * hdr_fps)

    # Window size in frames
    window_frames = int(_WINDOW_SECONDS * hdr_fps)
    window_frames = max(window_frames, 100)

    # Total duration
    hdr_duration = hdr_total / hdr_fps

    # Multi-window positions: distribute across 10%-90% of HDR timeline
    # Avoid first/last 10% (intros/credits)
    n_windows = _DEFAULT_N_WINDOWS
    margin_seconds = hdr_duration * 0.10
    available_start = margin_seconds
    available_end = hdr_duration - margin_seconds - _WINDOW_SECONDS

    if available_end <= available_start:
        # Source too short — use single centered window
        n_windows = 1
        available_start = max(0.0, hdr_duration / 2.0 - _WINDOW_SECONDS / 2.0)
        available_end = available_start

    if n_windows == 1:
        window_positions = [available_start]
    else:
        window_positions = np.linspace(
            available_start, available_end, n_windows
        ).tolist()

    logger.info(
        f"Multi-window sync: {n_windows} windows of {_WINDOW_SECONDS:.0f}s, "
        f"positions: {[f'{p:.1f}s' for p in window_positions]}"
    )

    # Estimate offset from each window
    results: list[tuple[int, float, float]] = []
    for i, pos in enumerate(window_positions):
        offset, score, peak_ratio = _estimate_offset_single_window(
            hdr_source, om_source,
            hdr_start_seconds=pos,
            window_frames=window_frames,
            max_offset_frames=max_offset_frames,
            proxy_width=config.proxy_width,
        )
        results.append((offset, score, peak_ratio))
        logger.debug(
            f"  Window {i + 1}: offset={offset}, score={score:.4f}, "
            f"peak_ratio={peak_ratio:.2f}"
        )

    # Consensus
    frame_offset, confidence = _multi_window_consensus(results, config.min_confidence)

    # Determine status
    if confidence >= config.min_confidence:
        status = SyncStatus.LOCKED
        frame_locked = True
    else:
        status = SyncStatus.FAILED
        frame_locked = False
        logger.warning(
            f"Sync confidence {confidence:.4f} below threshold "
            f"{config.min_confidence}. Status=FAILED."
        )

    offset_seconds = frame_offset / hdr_fps if hdr_fps > 0 else 0.0

    model = SyncModel(
        frame_offset=frame_offset,
        confidence=confidence,
        status=status,
        frame_locked=frame_locked,
        offset_seconds=offset_seconds,
        method="multi_window_temporal_edge_diff_ncc",
    )

    logger.info(
        f"Global offset: {frame_offset} frames ({offset_seconds:.3f}s), "
        f"confidence={confidence:.4f}, status={status.value}"
    )

    return model


# ---------------------------------------------------------------------------
# Public API: validate_sync
# ---------------------------------------------------------------------------


def _measure_local_offset_at_point(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    hdr_frame: int,
    global_offset: int,
    proxy_width: int = 480,
    window: int = 15,
    search_radius: int = 5,
) -> tuple[int, float]:
    """Measure the local offset at a specific HDR frame position.

    Uses batch extraction for the small local window (single FFmpeg process
    per source per checkpoint).

    Args:
        hdr_source: HDR source.
        om_source: Open Matte source.
        hdr_frame: Center frame on HDR timeline.
        global_offset: Expected global offset.
        proxy_width: Proxy resolution width.
        window: Half-window size in frames for local signal.
        search_radius: Search radius around expected offset.

    Returns:
        (local_offset, local_confidence)
    """
    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 24.0
    om_total = _get_total_frames(om_source)
    hdr_total = _get_total_frames(hdr_source)

    # HDR: extract 2*window frames centered on hdr_frame
    hdr_start_frame = max(0, hdr_frame - window)
    hdr_n_frames = min(2 * window, hdr_total - hdr_start_frame)
    hdr_start_seconds = hdr_start_frame / hdr_fps

    if hdr_n_frames < 10:
        return global_offset, 0.0

    # Expected OM position
    expected_om_frame = hdr_frame + global_offset
    om_start_frame = max(0, expected_om_frame - window - search_radius)
    om_n_frames = min(2 * window + 2 * search_radius, om_total - om_start_frame)
    om_start_seconds = om_start_frame / hdr_fps  # Use HDR fps for time mapping

    if om_n_frames < 10:
        return global_offset, 0.0

    # Batch extract and compute diff signals
    hdr_signal = _build_diff_signal_batch(
        hdr_source, hdr_start_seconds, hdr_n_frames, proxy_width=proxy_width
    )
    om_signal = _build_diff_signal_batch(
        om_source, om_start_seconds, om_n_frames, proxy_width=proxy_width
    )

    # Check activity
    if np.std(hdr_signal) < _MIN_DIFF_ENERGY or np.std(om_signal) < _MIN_DIFF_ENERGY:
        return global_offset, 0.5

    # Cross-correlate with limited search
    best_local_offset, score, _ = _find_best_offset_ncc(
        hdr_signal, om_signal, max_offset=search_radius + window
    )

    # Convert back to absolute offset
    local_offset = om_start_frame + best_local_offset - hdr_start_frame

    return local_offset, max(0.0, score)


def validate_sync(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    sync_model: SyncModel,
    config: SyncConfig | None = None,
) -> SyncModel:
    """Validate synchronization by checking offset consistency at multiple points.

    Samples `validation_points` checkpoints distributed across the HDR timeline.
    At each checkpoint, measures the local offset and compares with the global.
    If drift exceeds max_drift_frames at any checkpoint, marks as DRIFT_DETECTED.

    Args:
        hdr_source: Inspected HDR source.
        om_source: Inspected Open Matte source.
        sync_model: The sync model from find_global_offset.
        config: Synchronization configuration.

    Returns:
        Updated SyncModel with validation results.

    Raises:
        SyncDriftError: If drift exceeds the allowed threshold.
    """
    if config is None:
        config = SyncConfig()

    hdr_total = _get_total_frames(hdr_source)
    om_total = _get_total_frames(om_source)
    global_offset = sync_model.frame_offset

    n_points = config.validation_points

    # Distribute checkpoints across HDR timeline, avoiding first/last 5%
    margin = max(50, int(hdr_total * 0.05))
    available_start = margin
    available_end = hdr_total - margin

    if available_end <= available_start:
        # Source too short for meaningful validation — trust initial offset
        logger.warning("Source too short for multi-point validation.")
        sync_model.status = SyncStatus.LOCKED
        sync_model.frame_locked = True
        sync_model.drift_frames = 0.0
        return sync_model

    checkpoint_frames = np.linspace(
        available_start, available_end, n_points, dtype=int
    ).tolist()

    # Validate each checkpoint
    offsets: list[int] = []
    errors: list[float] = []
    checkpoints: list[dict] = []

    for cp_frame in checkpoint_frames:
        # Ensure the expected OM frame exists
        expected_om = cp_frame + global_offset
        if expected_om < 0 or expected_om >= om_total:
            continue

        local_offset, local_conf = _measure_local_offset_at_point(
            hdr_source, om_source, cp_frame, global_offset,
            proxy_width=config.proxy_width,
        )

        offset_error = abs(local_offset - global_offset)
        offsets.append(local_offset)
        errors.append(float(offset_error))
        checkpoints.append({
            "hdr_frame": cp_frame,
            "expected_om_frame": expected_om,
            "measured_offset": local_offset,
            "error_frames": float(offset_error),
            "confidence": local_conf,
        })

    if not errors:
        logger.warning("No valid checkpoints for validation.")
        sync_model.status = SyncStatus.FAILED
        sync_model.frame_locked = False
        return sync_model

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    drift = max_error  # Worst-case drift

    logger.info(
        f"Validation: {len(errors)} checkpoints, "
        f"mean_error={mean_error:.3f}, max_error={max_error:.3f} frames"
    )

    # Update model
    sync_model.mean_error_frames = mean_error
    sync_model.max_error_frames = max_error
    sync_model.drift_frames = drift
    sync_model.checkpoints = checkpoints

    # Check drift threshold
    if drift > config.max_drift_frames:
        sync_model.status = SyncStatus.DRIFT_DETECTED
        sync_model.frame_locked = False
        raise SyncDriftError(
            f"Temporal drift detected: max_error={max_error:.3f} frames "
            f"exceeds threshold {config.max_drift_frames}. "
            f"Synchronization is NOT frame-locked."
        )

    # Passed validation
    sync_model.status = SyncStatus.LOCKED
    sync_model.frame_locked = True

    logger.info(
        f"Sync VALIDATED: drift={drift:.3f} frames "
        f"(threshold={config.max_drift_frames})"
    )

    return sync_model
