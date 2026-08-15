"""Frame extraction and manipulation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from auto_openmatte.utils.ffmpeg import extract_frame_at_time, extract_frame_to_numpy


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
