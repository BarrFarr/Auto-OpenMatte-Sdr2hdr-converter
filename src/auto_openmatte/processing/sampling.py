"""Overlap sampling — extract paired HDR/SDR luminance from the overlap region.

This module bridges geometry/sync with the luminance estimator by:
1. Selecting representative frames from a shot.
2. Extracting the overlap region from both HDR and OM.
3. Linearizing both into their respective luminance spaces.
4. Rejecting low-quality pixels (black, clipped, near-zero).
5. Returning paired luminance arrays for curve fitting.

Color spaces:
- SDR input: BT.709 signal [0,1] → linearize via BT.1886 → linear RGB [0,1]
  → luminance Y = 0.2126R + 0.7152G + 0.0722B
- HDR input: PQ signal [0,1] → ST 2084 EOTF → absolute luminance [0,10000] cd/m²
  → normalized by peak_nits to [0,1] for curve fitting

The luminance curve maps: SDR_linear_Y → HDR_normalized_Y
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SourceInfo, SyncModel
from auto_openmatte.core.transfer_functions import linearize
from auto_openmatte.utils.ffmpeg import get_media_tool_config
from auto_openmatte.utils.frames import extract_segment_at_time_grayscale

logger = logging.getLogger(__name__)

# BT.709 luminance weights
_LUM_R = 0.2126
_LUM_G = 0.7152
_LUM_B = 0.0722

# BT.2020 luminance weights (for HDR per-channel linear luminance)
_LUM_R_2020 = 0.2627
_LUM_G_2020 = 0.6780
_LUM_B_2020 = 0.0593

# BT.709 → BT.2020 linear RGB conversion matrix (ITU-R BT.2087)
# Preserves luminance (Y709 ≈ Y2020 to < 0.004% precision)
_M_709_TO_2020 = np.array([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690972, 0.9195404, 0.0113624],
    [0.0163914, 0.0880133, 0.8955953],
], dtype=np.float64)

# Minimum luminance to keep (reject deep black)
_MIN_LUMINANCE = 0.001
# Maximum SDR value to keep (reject hard-clipped white)
_MAX_SDR_SIGNAL = 0.99
# Minimum SDR value to keep (reject crushed black)
_MIN_SDR_SIGNAL = 0.01


@dataclass
class OverlapSamples:
    """Paired luminance samples from the HDR/SDR overlap region."""

    # SDR linear luminance [0,1] — BT.1886 linearized
    sdr_luminance: NDArray[np.floating] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # HDR normalized luminance [0,1] — PQ→absolute→normalized by peak_nits
    hdr_luminance: NDArray[np.floating] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # Number of frames sampled
    n_frames: int = 0
    # Number of raw pixel pairs before rejection
    n_raw_pairs: int = 0
    # Number of valid pairs after rejection
    n_valid_pairs: int = 0
    # Rejection statistics
    n_rejected_black: int = 0
    n_rejected_clipped: int = 0
    n_rejected_nan: int = 0


@dataclass
class LuminanceDiagnostics:
    """Diagnostic statistics for luminance relationship."""

    # Percentiles of SDR luminance
    sdr_percentiles: dict[str, float] = field(default_factory=dict)
    # Percentiles of HDR luminance
    hdr_percentiles: dict[str, float] = field(default_factory=dict)
    # Fit quality
    train_mae: float = 0.0
    train_rmse: float = 0.0
    val_mae: float = 0.0
    val_rmse: float = 0.0
    # Sample info
    n_train: int = 0
    n_val: int = 0
    n_bins_valid: int = 0
    # Monotonicity
    monotonic: bool = True
    # Confidence
    confidence: float = 0.0


def _extract_frame_rgb(
    source: SourceInfo,
    time_seconds: float,
    width: int = 960,
) -> NDArray[np.floating] | None:
    """Extract a single frame as RGB float64 [0,1] via rgb48le.

    Uses a single FFmpeg process to decode one frame at the given timestamp
    and output as rgb48le (16-bit per channel, 3 channels).

    Returns:
        Array of shape (H, W, 3) normalized to [0,1], or None on failure.
    """
    import subprocess

    stream = source.selected_stream
    stream_index = stream.index if stream else 0

    cmd = [
        get_media_tool_config().ffmpeg_command, "-v", "quiet", "-nostdin",
        "-ss", f"{time_seconds:.6f}",
        "-i", str(source.path),
        "-map", f"0:v:{stream_index}",
        "-vf", f"scale={width}:-1",
        "-frames:v", "1",
        "-pix_fmt", "rgb48le",
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
        if result.returncode != 0 or not result.stdout:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    raw = result.stdout
    bytes_per_pixel = 6  # 3 channels × 2 bytes (uint16)
    total_pixels = len(raw) // bytes_per_pixel
    height = total_pixels // width

    if height <= 0 or width * height * bytes_per_pixel > len(raw):
        return None

    arr = np.frombuffer(raw[:width * height * bytes_per_pixel], dtype=np.uint16)
    return arr.reshape(height, width, 3).astype(np.float64) / 65535.0


def _extract_frame_grayscale(
    source: SourceInfo,
    time_seconds: float,
    width: int = 960,
) -> NDArray[np.floating] | None:
    """Extract a single frame as grayscale float64 [0,1] via gray16le.

    Used for SDR sources where grayscale Y' is acceptable for linearization.
    """
    stream = source.selected_stream
    stream_index = stream.index if stream else 0

    frames = extract_segment_at_time_grayscale(
        source.path, time_seconds, n_frames=1,
        stream_index=stream_index, width=width,
    )
    if not frames or frames[0] is None:
        return None
    return frames[0]


def sample_overlap_luminance(
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    sync_model: SyncModel,
    geometry: GeometryModel,
    shot: Shot,
    config: ColorConfig | None = None,
    n_samples: int = 30,
    proxy_width: int = 960,
    peak_nits: float = 10000.0,
    rng_seed: int = 42,
) -> OverlapSamples:
    """Sample paired luminance values from the overlap region for a shot.

    Extracts frames from both HDR and OM at corresponding positions,
    crops to the overlap region, linearizes, and collects paired luminance.

    Args:
        hdr_source: HDR source info.
        om_source: Open Matte source info.
        sync_model: Sync model (provides frame_offset).
        geometry: Geometry model (provides overlap_bbox).
        shot: Shot to sample from.
        config: Color configuration.
        n_samples: Number of frames to sample from the shot.
        proxy_width: Width for extraction (lower = faster, higher = more pixels).
        peak_nits: PQ peak luminance for normalization.
        rng_seed: Random seed for reproducibility.

    Returns:
        OverlapSamples with paired luminance arrays.
    """
    if config is None:
        config = ColorConfig()

    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 24.0
    offset = sync_model.frame_offset

    # Determine frame sampling positions within the shot
    shot_duration = shot.hdr_end_frame - shot.hdr_start_frame
    if shot_duration < 2:
        return OverlapSamples()

    n_samples = min(n_samples, shot_duration)
    rng = np.random.default_rng(rng_seed)

    # Sample frames quasi-randomly within the shot
    sample_frames = sorted(
        rng.choice(
            range(shot.hdr_start_frame, shot.hdr_end_frame),
            size=n_samples,
            replace=False,
        ).tolist()
    ) if shot_duration >= n_samples else list(
        range(shot.hdr_start_frame, shot.hdr_end_frame)
    )

    # Geometry: overlap region in OM coordinates (full-res)
    bbox = geometry.overlap_bbox
    # Convert bbox to proxy coordinates
    om_w = om_source.selected_stream.width if om_source.selected_stream else 3840
    scale_to_proxy = proxy_width / om_w
    proxy_x1 = int(round(bbox[0] * scale_to_proxy))
    proxy_y1 = int(round(bbox[1] * scale_to_proxy))
    proxy_x2 = int(round(bbox[2] * scale_to_proxy))
    proxy_y2 = int(round(bbox[3] * scale_to_proxy))

    all_sdr: list[NDArray] = []
    all_hdr: list[NDArray] = []
    n_raw = 0
    n_black = 0
    n_clipped = 0
    n_nan = 0
    n_frames_used = 0

    for hdr_frame in sample_frames:
        hdr_time = hdr_frame / hdr_fps
        om_frame = hdr_frame + offset
        om_time = om_frame / hdr_fps

        # Extract HDR as RGB (for correct per-channel PQ EOTF)
        hdr_rgb = _extract_frame_rgb(hdr_source, hdr_time, width=proxy_width)
        # Extract SDR as RGB (for correct per-channel BT.1886 EOTF)
        om_rgb = _extract_frame_rgb(om_source, om_time, width=proxy_width)

        if hdr_rgb is None or om_rgb is None:
            continue

        # Crop OM to overlap region
        om_h, om_w_actual = om_rgb.shape[:2]
        # Clamp proxy bbox
        py1 = max(0, min(proxy_y1, om_h))
        py2 = max(0, min(proxy_y2, om_h))
        px1 = max(0, min(proxy_x1, om_w_actual))
        px2 = max(0, min(proxy_x2, om_w_actual))

        if py2 <= py1 or px2 <= px1:
            continue

        om_overlap = om_rgb[py1:py2, px1:px2, :]

        # HDR: the entire frame corresponds to the overlap region
        # Resize HDR RGB to match OM overlap dimensions for pixel correspondence
        hdr_h, hdr_w_actual = hdr_rgb.shape[:2]
        target_h = py2 - py1
        target_w = px2 - px1

        if hdr_h != target_h or hdr_w_actual != target_w:
            from scipy.ndimage import zoom
            zoom_h = target_h / hdr_h
            zoom_w = target_w / hdr_w_actual
            hdr_resized = zoom(hdr_rgb, (zoom_h, zoom_w, 1.0), order=1)
        else:
            hdr_resized = hdr_rgb

        # Ensure same spatial shape
        min_h = min(hdr_resized.shape[0], om_overlap.shape[0])
        min_w = min(hdr_resized.shape[1], om_overlap.shape[1])
        hdr_crop = hdr_resized[:min_h, :min_w, :]
        om_crop = om_overlap[:min_h, :min_w, :]

        # --- Linearize SDR (per-channel BT.1886 EOTF + gamut convert + BT.2020 luminance) ---
        sdr_r_linear = linearize(om_crop[..., 0], "bt709")
        sdr_g_linear = linearize(om_crop[..., 1], "bt709")
        sdr_b_linear = linearize(om_crop[..., 2], "bt709")
        # Convert linear BT.709 RGB → linear BT.2020 RGB
        sdr_rgb_709 = np.stack([sdr_r_linear, sdr_g_linear, sdr_b_linear], axis=-1)
        shape = sdr_rgb_709.shape
        sdr_rgb_2020 = sdr_rgb_709.reshape(-1, 3) @ _M_709_TO_2020.T
        sdr_rgb_2020 = sdr_rgb_2020.reshape(shape)
        sdr_rgb_2020 = np.maximum(sdr_rgb_2020, 0.0)  # Clamp negatives from gamut
        # BT.2020 luminance
        sdr_linear = (
            _LUM_R_2020 * sdr_rgb_2020[..., 0]
            + _LUM_G_2020 * sdr_rgb_2020[..., 1]
            + _LUM_B_2020 * sdr_rgb_2020[..., 2]
        )

        # --- Linearize HDR (per-channel PQ EOTF + BT.2020 luminance) ---
        hdr_r_linear = linearize(hdr_crop[..., 0], "smpte2084", peak_nits=peak_nits)
        hdr_g_linear = linearize(hdr_crop[..., 1], "smpte2084", peak_nits=peak_nits)
        hdr_b_linear = linearize(hdr_crop[..., 2], "smpte2084", peak_nits=peak_nits)
        hdr_linear = (
            _LUM_R_2020 * hdr_r_linear
            + _LUM_G_2020 * hdr_g_linear
            + _LUM_B_2020 * hdr_b_linear
        )

        # Flatten
        sdr_flat = sdr_linear.flatten()
        hdr_flat = hdr_linear.flatten()
        n_raw += len(sdr_flat)

        # Rejection mask
        valid = np.ones(len(sdr_flat), dtype=bool)

        # Reject NaN/Inf
        nan_mask = ~(np.isfinite(sdr_flat) & np.isfinite(hdr_flat))
        n_nan += int(np.sum(nan_mask))
        valid &= ~nan_mask

        # Reject deep black (both sources)
        black_mask = (sdr_flat < _MIN_LUMINANCE) & (hdr_flat < _MIN_LUMINANCE)
        n_black += int(np.sum(black_mask))
        valid &= ~black_mask

        # Reject clipped SDR
        # Use SDR signal-domain luminance (Y') for clipping check
        om_signal_y = (
            _LUM_R * om_crop[..., 0]
            + _LUM_G * om_crop[..., 1]
            + _LUM_B * om_crop[..., 2]
        ).flatten()
        clip_mask = (om_signal_y > _MAX_SDR_SIGNAL) | (om_signal_y < _MIN_SDR_SIGNAL)
        n_clipped += int(np.sum(clip_mask))
        valid &= ~clip_mask

        if np.sum(valid) > 0:
            all_sdr.append(sdr_flat[valid])
            all_hdr.append(hdr_flat[valid])
            n_frames_used += 1

    # Concatenate
    if all_sdr:
        sdr_arr = np.concatenate(all_sdr)
        hdr_arr = np.concatenate(all_hdr)
    else:
        sdr_arr = np.array([], dtype=np.float64)
        hdr_arr = np.array([], dtype=np.float64)

    samples = OverlapSamples(
        sdr_luminance=sdr_arr,
        hdr_luminance=hdr_arr,
        n_frames=n_frames_used,
        n_raw_pairs=n_raw,
        n_valid_pairs=len(sdr_arr),
        n_rejected_black=n_black,
        n_rejected_clipped=n_clipped,
        n_rejected_nan=n_nan,
    )

    logger.info(
        f"Shot {shot.shot_id}: sampled {n_frames_used}/{len(sample_frames)} frames, "
        f"{samples.n_valid_pairs}/{n_raw} valid pairs "
        f"(rejected: black={n_black}, clipped={n_clipped}, nan={n_nan})"
    )

    return samples


def train_validation_split(
    samples: OverlapSamples,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[OverlapSamples, OverlapSamples]:
    """Split overlap samples into train and validation sets.

    Args:
        samples: Full overlap samples.
        train_ratio: Fraction for training (default 80%).
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, val_samples)
    """
    n = samples.n_valid_pairs
    if n < 10:
        return samples, OverlapSamples()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split = int(n * train_ratio)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train = OverlapSamples(
        sdr_luminance=samples.sdr_luminance[train_idx],
        hdr_luminance=samples.hdr_luminance[train_idx],
        n_frames=samples.n_frames,
        n_raw_pairs=samples.n_raw_pairs,
        n_valid_pairs=len(train_idx),
    )
    val = OverlapSamples(
        sdr_luminance=samples.sdr_luminance[val_idx],
        hdr_luminance=samples.hdr_luminance[val_idx],
        n_frames=samples.n_frames,
        n_raw_pairs=samples.n_raw_pairs,
        n_valid_pairs=len(val_idx),
    )
    return train, val


def compute_diagnostics(
    samples: OverlapSamples,
    curve: list[list[float]],
    train_samples: OverlapSamples | None = None,
    val_samples: OverlapSamples | None = None,
) -> LuminanceDiagnostics:
    """Compute diagnostic statistics for a luminance transform.

    Args:
        samples: Full overlap samples.
        curve: Fitted luminance curve control points.
        train_samples: Training subset (for train error).
        val_samples: Validation subset (for val error).

    Returns:
        LuminanceDiagnostics with percentiles, errors, and quality info.
    """
    from auto_openmatte.processing.luminance import apply_luminance_curve

    diag = LuminanceDiagnostics()

    if samples.n_valid_pairs == 0:
        return diag

    sdr = samples.sdr_luminance
    hdr = samples.hdr_luminance

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        diag.sdr_percentiles[f"P{p}"] = float(np.percentile(sdr, p))
        diag.hdr_percentiles[f"P{p}"] = float(np.percentile(hdr, p))

    # Check monotonicity of curve
    if curve and len(curve) >= 2:
        for i in range(1, len(curve)):
            if curve[i][1] < curve[i - 1][1] - 1e-10:
                diag.monotonic = False
                break

    # Train error
    if train_samples and train_samples.n_valid_pairs > 0:
        mapped_train = apply_luminance_curve(train_samples.sdr_luminance, curve)
        errors_train = np.abs(mapped_train - train_samples.hdr_luminance)
        diag.train_mae = float(np.mean(errors_train))
        diag.train_rmse = float(np.sqrt(np.mean(errors_train ** 2)))
        diag.n_train = train_samples.n_valid_pairs

    # Validation error
    if val_samples and val_samples.n_valid_pairs > 0:
        mapped_val = apply_luminance_curve(val_samples.sdr_luminance, curve)
        errors_val = np.abs(mapped_val - val_samples.hdr_luminance)
        diag.val_mae = float(np.mean(errors_val))
        diag.val_rmse = float(np.sqrt(np.mean(errors_val ** 2)))
        diag.n_val = val_samples.n_valid_pairs

    # Confidence from correlation
    mapped_all = apply_luminance_curve(sdr, curve)
    if np.std(mapped_all) > 0 and np.std(hdr) > 0:
        corr = float(np.corrcoef(mapped_all, hdr)[0, 1])
        diag.confidence = max(0.0, corr)

    return diag
