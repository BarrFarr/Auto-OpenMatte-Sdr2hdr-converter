"""Shot detection on the HDR timeline.

Detects shot boundaries (hard cuts, fades, dissolves) by analyzing
frame-to-frame differences using multiple metrics:
1. Luminance histogram difference (chi-square)
2. Mean luminance absolute difference
3. Edge correlation change (1 - NCC of consecutive edge maps)

Processing is done on low-resolution grayscale proxy (~480px) extracted
via batch FFmpeg (single subprocess per segment). Frames are processed
in a streaming fashion to minimize memory usage.

HDR is the only reference timeline. Open Matte boundaries are derived
by adding sync_model.frame_offset to each HDR boundary.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import ShotConfig
from auto_openmatte.core.models import Shot, SourceInfo, SyncModel
from auto_openmatte.utils.frames import compute_edge_map, extract_segment_at_time_grayscale
from auto_openmatte.utils.math_utils import normalized_cross_correlation

logger = logging.getLogger(__name__)

# Number of histogram bins for luminance histogram difference
_HIST_BINS = 64

# Segment size for batch extraction (frames per FFmpeg call)
_SEGMENT_FRAMES = 500

# Minimum edge energy to consider a frame "informative" (not black/blank)
_MIN_FRAME_ENERGY = 0.005

# Number of consecutive frames to detect gradual transitions
_GRADUAL_MIN_FRAMES = 5

# Weights for combined boundary score
_WEIGHT_HISTOGRAM = 0.4
_WEIGHT_LUMINANCE = 0.2
_WEIGHT_EDGE = 0.4


# ---------------------------------------------------------------------------
# Frame metrics
# ---------------------------------------------------------------------------


def _compute_histogram(frame: NDArray[np.floating], n_bins: int = _HIST_BINS) -> NDArray:
    """Compute normalized luminance histogram of a grayscale frame."""
    hist, _ = np.histogram(frame, bins=n_bins, range=(0.0, 1.0))
    total = hist.sum()
    if total > 0:
        return hist.astype(np.float64) / total
    return hist.astype(np.float64)


def _histogram_chi_square(hist_a: NDArray, hist_b: NDArray) -> float:
    """Chi-square distance between two normalized histograms.

    Returns a value in [0, 2] where 0 = identical, 2 = maximally different.
    """
    denom = hist_a + hist_b
    # Avoid division by zero
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    diff_sq = (hist_a[mask] - hist_b[mask]) ** 2
    chi2 = float(np.sum(diff_sq / denom[mask]))
    return chi2


def _mean_luminance(frame: NDArray[np.floating]) -> float:
    """Mean luminance of a frame (already normalized to [0,1])."""
    return float(np.mean(frame))


def _edge_ncc(edge_a: NDArray[np.floating], edge_b: NDArray[np.floating]) -> float:
    """Normalized cross-correlation between two edge maps.

    Returns value in [-1, 1] where 1 = identical structure.
    """
    # Ensure same shape via center crop
    min_h = min(edge_a.shape[0], edge_b.shape[0])
    min_w = min(edge_a.shape[1], edge_b.shape[1])
    if min_h < 2 or min_w < 2:
        return 0.0
    a = edge_a[:min_h, :min_w]
    b = edge_b[:min_h, :min_w]
    return normalized_cross_correlation(a, b)


def compute_boundary_score(
    frame_a: NDArray[np.floating],
    frame_b: NDArray[np.floating],
    edge_a: NDArray[np.floating],
    edge_b: NDArray[np.floating],
) -> tuple[float, float, float, float]:
    """Compute combined boundary score between two consecutive frames.

    Returns:
        (combined_score, hist_score, lum_score, edge_score)
        All scores in [0, 1] range (normalized).
    """
    # 1. Histogram chi-square (range [0, 2] → normalize to [0, 1])
    hist_a = _compute_histogram(frame_a)
    hist_b = _compute_histogram(frame_b)
    hist_score = min(1.0, _histogram_chi_square(hist_a, hist_b) / 2.0)

    # 2. Mean luminance absolute difference (range [0, 1])
    lum_score = abs(_mean_luminance(frame_a) - _mean_luminance(frame_b))

    # 3. Edge correlation change (range [0, 1], where 1 = completely different)
    ncc = _edge_ncc(edge_a, edge_b)
    edge_score = max(0.0, 1.0 - ncc)  # 0 = same, 1 = completely different

    # Combined weighted score
    combined = (
        _WEIGHT_HISTOGRAM * hist_score
        + _WEIGHT_LUMINANCE * lum_score
        + _WEIGHT_EDGE * edge_score
    )

    return combined, hist_score, lum_score, edge_score


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------


def _compute_adaptive_threshold(
    scores: NDArray[np.floating],
    multiplier: float = 3.0,
) -> float:
    """Compute adaptive threshold using robust statistics.

    Excludes top 5% of scores (potential cuts) when computing baseline
    mean and std, so genuine cuts don't inflate the threshold.

    Args:
        scores: 1-D array of boundary scores.
        multiplier: Number of standard deviations above mean.

    Returns:
        Threshold value.
    """
    if len(scores) == 0:
        return 1.0

    # Exclude top 5% to avoid real cuts inflating the baseline
    cutoff = np.percentile(scores, 95)
    baseline = scores[scores <= cutoff]

    if len(baseline) < 5:
        # Too few samples — use all scores
        baseline = scores

    mean = float(np.mean(baseline))
    std = float(np.std(baseline))

    # Ensure minimum threshold to avoid false positives in very static content
    threshold = mean + multiplier * std
    return max(threshold, 0.02)


# ---------------------------------------------------------------------------
# Transition classification
# ---------------------------------------------------------------------------


def _is_low_information_boundary(
    frame_a: NDArray[np.floating],
    frame_b: NDArray[np.floating],
    edge_a: NDArray[np.floating],
    edge_b: NDArray[np.floating],
) -> bool:
    """Check if a boundary is between two low-information frames (black-to-black).

    Uses raw luminance level and edge variance (not normalized magnitude)
    to detect truly uninformative frames.
    """
    lum_a = _mean_luminance(frame_a)
    lum_b = _mean_luminance(frame_b)

    # Both frames are very dark → likely black/near-black
    if lum_a < 0.03 and lum_b < 0.03:
        return True

    # Both frames have extremely low variance (uniform/flat)
    var_a = float(np.var(frame_a))
    var_b = float(np.var(frame_b))
    if var_a < 0.0001 and var_b < 0.0001:
        return True

    return False


def _classify_transitions(
    scores: NDArray[np.floating],
    threshold: float,
    mean_luminances: NDArray[np.floating],
    detect_transitions: bool = True,
    min_shot_frames: int = 6,
) -> list[tuple[int, str, float]]:
    """Classify detected boundaries into hard cuts, fades, and dissolves.

    Args:
        scores: 1-D array of combined boundary scores.
        threshold: Adaptive threshold for hard cut detection.
        mean_luminances: Per-frame mean luminance (for fade detection).
        detect_transitions: Whether to detect fades/dissolves.
        min_shot_frames: Minimum shot duration.

    Returns:
        List of (frame_index, cut_type, confidence) for each detected boundary.
        frame_index is the index of the SECOND frame in the transition
        (i.e., the first frame of the new shot).
    """
    n = len(scores)
    if n == 0:
        return []

    boundaries: list[tuple[int, str, float]] = []
    gradual_threshold = threshold * 0.5

    # Mark which frames are part of gradual transitions (to avoid double-counting)
    used = np.zeros(n, dtype=bool)

    # --- Pass 1: Hard cuts (single frame spikes) ---
    for i in range(n):
        if scores[i] >= threshold and not used[i]:
            # Check it's a spike (not part of gradual transition)
            # A hard cut has high score at frame i but lower at i-1 and i+1
            is_spike = True
            if i > 0 and scores[i - 1] >= gradual_threshold:
                is_spike = False
            if i < n - 1 and scores[i + 1] >= gradual_threshold:
                is_spike = False

            if is_spike:
                confidence = min(1.0, scores[i] / threshold)
                boundaries.append((i + 1, "hard", confidence))  # +1: new shot starts here
                used[i] = True

    # --- Pass 2: Gradual transitions (fades and dissolves) ---
    if detect_transitions:
        i = 0
        while i < n:
            if used[i] or scores[i] < gradual_threshold:
                i += 1
                continue

            # Find run of elevated scores
            run_start = i
            while i < n and scores[i] >= gradual_threshold and not used[i]:
                i += 1
            run_end = i

            run_length = run_end - run_start
            if run_length < _GRADUAL_MIN_FRAMES:
                continue

            # Classify: fade vs dissolve
            # Fade: luminance trends toward near-zero (fade-out) or from near-zero (fade-in)
            lum_start = mean_luminances[run_start] if run_start < len(mean_luminances) else 0.5
            lum_end = mean_luminances[min(run_end, len(mean_luminances) - 1)]

            if lum_end < 0.05 or lum_start < 0.05:
                cut_type = "fade"
            else:
                cut_type = "dissolve"

            # Boundary at the middle of the transition
            mid = run_start + run_length // 2 + 1
            confidence = min(1.0, float(np.max(scores[run_start:run_end])) / threshold)
            boundaries.append((mid, cut_type, confidence))

            # Mark used
            used[run_start:run_end] = True

    # Sort by frame index
    boundaries.sort(key=lambda x: x[0])

    # --- Pass 3: Enforce min_shot_frames ---
    if len(boundaries) < 2:
        return boundaries

    filtered: list[tuple[int, str, float]] = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary[0] - filtered[-1][0] >= min_shot_frames:
            filtered.append(boundary)
        else:
            # Keep the one with higher confidence
            if boundary[2] > filtered[-1][2]:
                filtered[-1] = boundary

    return filtered


# ---------------------------------------------------------------------------
# Main extraction + scoring loop
# ---------------------------------------------------------------------------


def _extract_and_score_segment(
    hdr_source: SourceInfo,
    start_seconds: float,
    n_frames: int,
    proxy_width: int = 480,
    prev_frame: NDArray[np.floating] | None = None,
    prev_edge: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
]:
    """Extract a segment and compute boundary scores.

    Processes frames streaming-style: computes metrics between consecutive
    frames and discards data as it goes.

    Args:
        hdr_source: HDR source info.
        start_seconds: Start time for extraction.
        n_frames: Number of frames to extract.
        proxy_width: Proxy resolution.
        prev_frame: Last frame from previous segment (for continuity).
        prev_edge: Last edge map from previous segment.

    Returns:
        (scores, mean_luminances, last_frame, last_edge)
        scores: 1-D array of boundary scores (length = n_frames - 1, or n_frames if prev provided)
        mean_luminances: per-frame mean luminance
        last_frame: last valid frame (for next segment continuity)
        last_edge: last valid edge map
    """
    stream = hdr_source.selected_stream
    stream_index = stream.index if stream else 0

    frames = extract_segment_at_time_grayscale(
        hdr_source.path,
        start_seconds,
        n_frames,
        stream_index=stream_index,
        width=proxy_width,
    )

    # Compute scores streaming
    n_boundaries = n_frames - 1 + (1 if prev_frame is not None else 0)
    scores = np.zeros(n_boundaries, dtype=np.float64)
    mean_lums = np.zeros(n_frames, dtype=np.float64)

    score_idx = 0
    last_valid_frame: NDArray[np.floating] | None = prev_frame
    last_valid_edge: NDArray[np.floating] | None = prev_edge

    for i, frame in enumerate(frames):
        if frame is None:
            last_valid_frame = None
            last_valid_edge = None
            continue

        edge = compute_edge_map(frame)
        mean_lums[i] = _mean_luminance(frame)

        if last_valid_frame is not None and last_valid_edge is not None:
            # Check low-information suppression
            if _is_low_information_boundary(last_valid_frame, frame, last_valid_edge, edge):
                scores[score_idx] = 0.0
            else:
                combined, _, _, _ = compute_boundary_score(
                    last_valid_frame, frame, last_valid_edge, edge
                )
                scores[score_idx] = combined
            score_idx += 1

        last_valid_frame = frame
        last_valid_edge = edge

    # Trim scores to actual computed length
    scores = scores[:score_idx]

    return scores, mean_lums, last_valid_frame, last_valid_edge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_shots(
    hdr_source: SourceInfo,
    sync_model: SyncModel,
    config: ShotConfig | None = None,
    frame_range: tuple[int, int] | None = None,
) -> list[Shot]:
    """Detect shot boundaries on the HDR timeline.

    Analyzes frame-to-frame differences using histogram, luminance, and
    edge-based metrics. Classifies boundaries as hard cuts, fades, or
    dissolves. Maps all boundaries to Open Matte via sync offset.

    Processing uses batch FFmpeg extraction (~500 frames per subprocess)
    with streaming metric computation to minimize memory.

    Args:
        hdr_source: Inspected HDR source with selected_stream.
        sync_model: Validated sync model (provides frame_offset).
        config: Shot detection configuration.
        frame_range: Optional (start_frame, end_frame) to limit detection.

    Returns:
        List of Shot objects sorted by hdr_start_frame.
    """
    if config is None:
        config = ShotConfig()

    stream = hdr_source.selected_stream
    if stream is None:
        return []

    fps = stream.fps if stream.fps > 0 else 24.0
    total_frames = stream.frame_count or int(stream.duration_seconds * fps)

    # Determine processing range
    range_start = 0
    range_end = total_frames
    if frame_range:
        range_start, range_end = frame_range
        range_start = max(0, range_start)
        range_end = min(total_frames, range_end)

    n_range_frames = range_end - range_start
    if n_range_frames < 2:
        return []

    start_seconds = range_start / fps
    offset = sync_model.frame_offset

    logger.info(
        f"Shot detection: frames {range_start}–{range_end} "
        f"({n_range_frames} frames, {n_range_frames / fps:.1f}s)"
    )

    # Process in segments with streaming metrics
    all_scores: list[float] = []
    all_mean_lums: list[float] = []

    frames_remaining = n_range_frames
    current_seconds = start_seconds
    prev_frame: NDArray[np.floating] | None = None
    prev_edge: NDArray[np.floating] | None = None
    segment_count = 0

    while frames_remaining > 0:
        seg_size = min(_SEGMENT_FRAMES, frames_remaining)

        scores, mean_lums, prev_frame, prev_edge = _extract_and_score_segment(
            hdr_source,
            start_seconds=current_seconds,
            n_frames=seg_size,
            proxy_width=480,
            prev_frame=prev_frame,
            prev_edge=prev_edge,
        )

        all_scores.extend(scores.tolist())
        all_mean_lums.extend(mean_lums.tolist())

        frames_remaining -= seg_size
        current_seconds += seg_size / fps
        segment_count += 1

    logger.info(f"Shot detection: {segment_count} segments processed")

    if not all_scores:
        # No boundaries could be computed — return single shot
        return [_make_single_shot(range_start, range_end, offset)]

    scores_arr = np.array(all_scores, dtype=np.float64)
    lums_arr = np.array(all_mean_lums, dtype=np.float64)

    # Compute adaptive threshold
    threshold = _compute_adaptive_threshold(scores_arr, multiplier=config.threshold_multiplier)
    logger.info(
        f"Shot detection: threshold={threshold:.4f}, "
        f"scores mean={np.mean(scores_arr):.4f}, std={np.std(scores_arr):.4f}, "
        f"max={np.max(scores_arr):.4f}"
    )

    # Classify transitions
    boundaries = _classify_transitions(
        scores_arr,
        threshold,
        lums_arr,
        detect_transitions=config.detect_transitions,
        min_shot_frames=config.min_shot_frames,
    )

    logger.info(f"Shot detection: {len(boundaries)} boundaries detected")

    if not boundaries:
        return [_make_single_shot(range_start, range_end, offset)]

    # Build Shot list from boundaries
    shots: list[Shot] = []
    shot_id = 0

    # First shot: from range_start to first boundary
    first_boundary_frame = range_start + boundaries[0][0]
    if first_boundary_frame > range_start:
        shots.append(Shot(
            shot_id=shot_id,
            hdr_start_frame=range_start,
            hdr_end_frame=first_boundary_frame,
            om_start_frame=range_start + offset,
            om_end_frame=first_boundary_frame + offset,
            duration_frames=first_boundary_frame - range_start,
            cut_type=boundaries[0][1],
            confidence=boundaries[0][2],
        ))
        shot_id += 1

    # Middle shots: between consecutive boundaries
    for i in range(len(boundaries) - 1):
        b_start = range_start + boundaries[i][0]
        b_end = range_start + boundaries[i + 1][0]
        shots.append(Shot(
            shot_id=shot_id,
            hdr_start_frame=b_start,
            hdr_end_frame=b_end,
            om_start_frame=b_start + offset,
            om_end_frame=b_end + offset,
            duration_frames=b_end - b_start,
            cut_type=boundaries[i + 1][1],
            confidence=boundaries[i + 1][2],
        ))
        shot_id += 1

    # Last shot: from last boundary to range_end
    last_boundary_frame = range_start + boundaries[-1][0]
    if last_boundary_frame < range_end:
        shots.append(Shot(
            shot_id=shot_id,
            hdr_start_frame=last_boundary_frame,
            hdr_end_frame=range_end,
            om_start_frame=last_boundary_frame + offset,
            om_end_frame=range_end + offset,
            duration_frames=range_end - last_boundary_frame,
            cut_type="hard",  # End of range — type unknown
            confidence=1.0,
        ))

    logger.info(f"Shot detection: {len(shots)} shots created")
    return shots


def _make_single_shot(start: int, end: int, offset: int) -> Shot:
    """Create a single shot covering the entire range."""
    return Shot(
        shot_id=0,
        hdr_start_frame=start,
        hdr_end_frame=end,
        om_start_frame=start + offset,
        om_end_frame=end + offset,
        duration_frames=end - start,
        cut_type="hard",
        confidence=1.0,
    )
