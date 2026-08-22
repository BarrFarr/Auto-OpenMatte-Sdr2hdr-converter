"""Frame extraction and manipulation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.utils.ffmpeg import (
    extract_frame_at_time,
    extract_frame_to_numpy,
    get_media_tool_config,
)


def frame_to_array(
    raw_data: bytes, width: int, height: int, dtype: str = "uint16"
) -> NDArray[np.floating] | None:
    """Convert raw pixel bytes to numpy array.

    Args:
        raw_data: Raw pixel data from FFmpeg.
        width: Frame width in pixels.
        height: Frame height in pixels.
        dtype: Numpy dtype for the data.

    Returns:
        2D numpy array (height, width) or None if data is invalid.
    """
    if not raw_data:
        return None
    expected_bytes = width * height * np.dtype(dtype).itemsize
    if len(raw_data) < expected_bytes:
        return None
    arr = np.frombuffer(raw_data[:expected_bytes], dtype=dtype)
    return arr.reshape(height, width).astype(np.float64)


def get_frame_grayscale(
    file_path: Path,
    frame_number: int,
    stream_index: int = 0,
    width: int = 480,
) -> NDArray[np.floating] | None:
    """Extract a single frame as a grayscale float64 array.

    Uses 16-bit grayscale extraction for precision.
    Returns values normalized to [0, 1].

    Args:
        file_path: Path to video file.
        frame_number: Frame index (0-based).
        stream_index: Video stream index.
        width: Target width (height auto-calculated).

    Returns:
        Grayscale frame as float64 array normalized to [0, 1], or None on failure.
    """
    raw = extract_frame_to_numpy(
        file_path, frame_number, stream_index=stream_index, width=width, pix_fmt="gray16le"
    )
    if not raw:
        return None

    # Calculate height from raw data size
    bytes_per_pixel = 2  # 16-bit
    total_pixels = len(raw) // bytes_per_pixel
    height = total_pixels // width

    if height <= 0 or width * height * bytes_per_pixel > len(raw):
        return None

    arr = np.frombuffer(raw[: width * height * bytes_per_pixel], dtype=np.uint16)
    arr = arr.reshape(height, width).astype(np.float64)
    return arr / 65535.0


def get_frame_at_time_grayscale(
    file_path: Path,
    time_seconds: float,
    stream_index: int = 0,
    width: int = 480,
) -> NDArray[np.floating] | None:
    """Extract a frame at a timestamp as grayscale float64.

    Args:
        file_path: Path to video file.
        time_seconds: Timestamp in seconds.
        stream_index: Video stream index.
        width: Target width.

    Returns:
        Grayscale frame normalized to [0, 1], or None on failure.
    """
    raw = extract_frame_at_time(
        file_path, time_seconds, stream_index=stream_index, width=width, pix_fmt="gray16le"
    )
    if not raw:
        return None

    bytes_per_pixel = 2
    total_pixels = len(raw) // bytes_per_pixel
    height = total_pixels // width

    if height <= 0 or width * height * bytes_per_pixel > len(raw):
        return None

    arr = np.frombuffer(raw[: width * height * bytes_per_pixel], dtype=np.uint16)
    arr = arr.reshape(height, width).astype(np.float64)
    return arr / 65535.0


def compute_edge_map(frame: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute Sobel edge magnitude map.

    Edge maps are transfer-function agnostic — they capture structure
    regardless of whether the source is PQ, HLG, or gamma.

    Args:
        frame: 2D float array (grayscale).

    Returns:
        Edge magnitude map (same shape), normalized to [0, 1].
    """
    # Sobel kernels
    # X gradient
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    # Y gradient
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    from scipy.ndimage import convolve

    gx = convolve(frame, sobel_x, mode="reflect")
    gy = convolve(frame, sobel_y, mode="reflect")
    magnitude = np.sqrt(gx**2 + gy**2)

    # Normalize to [0, 1]
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val
    return magnitude


def extract_segment_grayscale(
    file_path: Path,
    start_frame: int,
    n_frames: int,
    stream_index: int = 0,
    width: int = 480,
    timeout: int = 300,
) -> list[NDArray[np.floating] | None]:
    """Extract a contiguous segment of frames as grayscale arrays using a SINGLE FFmpeg process.

    This is dramatically faster than per-frame extraction because FFmpeg only
    opens and seeks the file once, then decodes frames sequentially.

    The output is a list of float64 arrays normalized to [0, 1], or None for
    frames that could not be decoded.

    Args:
        file_path: Path to video file.
        start_frame: First frame index (0-based).
        n_frames: Number of consecutive frames to extract.
        stream_index: Video stream index.
        width: Target width (height auto-computed by aspect ratio).
        timeout: Maximum time in seconds for the FFmpeg process.

    Returns:
        List of length n_frames. Each element is either a float64 array
        of shape (H, W) normalized to [0,1], or None if that frame failed.
    """
    if n_frames <= 0:
        return []

    # Build FFmpeg command that:
    # 1. Seeks to start_frame via select filter
    # 2. Outputs n_frames of gray16le raw video at target width
    # 3. Pipes all frames as a single contiguous byte stream
    vf_parts = [
        f"select='gte(n\\,{start_frame})*lte(n\\,{start_frame + n_frames - 1})'",
        f"scale={width}:-1",
    ]

    cmd = [
        get_media_tool_config().ffmpeg_command,
        "-v", "quiet",
        "-nostdin",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", ",".join(vf_parts),
        "-vsync", "passthrough",
        "-frames:v", str(n_frames),
        "-pix_fmt", "gray16le",
        "-f", "rawvideo",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False
        )
        if result.returncode != 0:
            return [None] * n_frames
        raw = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return [None] * n_frames

    if not raw:
        return [None] * n_frames

    # Determine frame dimensions from total byte count
    bytes_per_pixel = 2  # gray16le
    total_pixels = len(raw) // bytes_per_pixel

    # We know width; compute height from first frame
    # All frames have the same dimensions
    if total_pixels < width:
        return [None] * n_frames

    # height = total_pixels / (width * actual_n_frames)
    # But we may have gotten fewer frames than requested
    # Try to find height by dividing total pixels by width and n_frames
    pixels_per_frame = total_pixels // n_frames if n_frames > 0 else 0
    if pixels_per_frame < width:
        # Fewer frames decoded than expected — recompute
        # Assume all decoded frames have same height
        # Try height = total_pixels / (width * actual_frames)
        # Start by guessing height from aspect ratio of common resolutions
        height = pixels_per_frame // width if width > 0 else 0
        if height <= 0:
            return [None] * n_frames
    else:
        height = pixels_per_frame // width

    if height <= 0:
        return [None] * n_frames

    frame_bytes = width * height * bytes_per_pixel
    actual_frames = len(raw) // frame_bytes

    frames: list[NDArray[np.floating] | None] = []
    for i in range(n_frames):
        if i < actual_frames:
            offset = i * frame_bytes
            frame_data = raw[offset: offset + frame_bytes]
            if len(frame_data) == frame_bytes:
                arr = np.frombuffer(frame_data, dtype=np.uint16)
                arr = arr.reshape(height, width).astype(np.float64) / 65535.0
                frames.append(arr)
            else:
                frames.append(None)
        else:
            frames.append(None)

    return frames


def extract_segment_at_time_grayscale(
    file_path: Path,
    start_seconds: float,
    n_frames: int,
    stream_index: int = 0,
    width: int = 480,
    timeout: int = 300,
) -> list[NDArray[np.floating] | None]:
    """Extract a contiguous segment starting at a timestamp using a SINGLE FFmpeg process.

    Uses input seeking (-ss before -i) for fast seeking, then decodes n_frames
    sequentially. Much faster than per-frame extraction.

    Args:
        file_path: Path to video file.
        start_seconds: Start timestamp in seconds.
        n_frames: Number of consecutive frames to extract.
        stream_index: Video stream index.
        width: Target width.
        timeout: Maximum time in seconds for the FFmpeg process.

    Returns:
        List of length n_frames with grayscale float64 arrays or None.
    """
    if n_frames <= 0:
        return []

    cmd = [
        get_media_tool_config().ffmpeg_command,
        "-v", "quiet",
        "-nostdin",
        "-ss", f"{start_seconds:.6f}",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", f"scale={width}:-1",
        "-frames:v", str(n_frames),
        "-pix_fmt", "gray16le",
        "-f", "rawvideo",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False
        )
        if result.returncode != 0:
            return [None] * n_frames
        raw = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return [None] * n_frames

    if not raw:
        return [None] * n_frames

    bytes_per_pixel = 2
    total_pixels = len(raw) // bytes_per_pixel

    if total_pixels < width:
        return [None] * n_frames

    pixels_per_frame = total_pixels // n_frames if n_frames > 0 else 0
    if pixels_per_frame < width:
        height = total_pixels // width
        if height <= 0:
            return [None] * n_frames
        # Recompute actual frame count
        pixels_per_frame = width * height
    else:
        height = pixels_per_frame // width

    if height <= 0:
        return [None] * n_frames

    frame_bytes = width * height * bytes_per_pixel
    actual_frames = len(raw) // frame_bytes

    frames: list[NDArray[np.floating] | None] = []
    for i in range(n_frames):
        if i < actual_frames:
            offset = i * frame_bytes
            frame_data = raw[offset: offset + frame_bytes]
            if len(frame_data) == frame_bytes:
                arr = np.frombuffer(frame_data, dtype=np.uint16)
                arr = arr.reshape(height, width).astype(np.float64) / 65535.0
                frames.append(arr)
            else:
                frames.append(None)
        else:
            frames.append(None)

    return frames
