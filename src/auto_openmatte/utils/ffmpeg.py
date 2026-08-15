"""FFmpeg and ffprobe subprocess helpers.

All operations explicitly select video streams and ignore audio.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from auto_openmatte.core.exceptions import InspectionError


def run_ffprobe(file_path: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output.

    Args:
        file_path: Path to the media file.

    Returns:
        Parsed JSON dict with format and stream information.

    Raises:
        InspectionError: If ffprobe fails or returns invalid JSON.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_frames",
        "-read_intervals", "%+#1",  # Only read first frame for side_data
        "-select_streams", "v",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise InspectionError(
                f"ffprobe failed for {file_path}: {result.stderr.strip()}"
            )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired as e:
        raise InspectionError(f"ffprobe timed out for {file_path}") from e
    except json.JSONDecodeError as e:
        raise InspectionError(f"ffprobe returned invalid JSON for {file_path}") from e
    except FileNotFoundError as e:
        raise InspectionError(
            "ffprobe not found. Ensure FFmpeg is installed and on PATH."
        ) from e


def run_ffprobe_streams_only(file_path: Path) -> dict[str, Any]:
    """Run ffprobe for stream info only (faster, no frame decoding).

    Args:
        file_path: Path to the media file.

    Returns:
        Parsed JSON dict with stream information.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise InspectionError(
                f"ffprobe failed for {file_path}: {result.stderr.strip()}"
            )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired as e:
        raise InspectionError(f"ffprobe timed out for {file_path}") from e
    except json.JSONDecodeError as e:
        raise InspectionError(f"ffprobe returned invalid JSON for {file_path}") from e
    except FileNotFoundError as e:
        raise InspectionError(
            "ffprobe not found. Ensure FFmpeg is installed and on PATH."
        ) from e


def extract_frame_to_numpy(
    file_path: Path,
    frame_number: int,
    stream_index: int = 0,
    width: int | None = None,
    pix_fmt: str = "gray16le",
) -> bytes:
    """Extract a single frame as raw pixel data using FFmpeg.

    Args:
        file_path: Path to the media file.
        frame_number: Frame number to extract (0-indexed).
        stream_index: Video stream index.
        width: Target width for scaling (None = original).
        pix_fmt: Output pixel format.

    Returns:
        Raw pixel data as bytes.
    """
    vf_parts = []
    if width:
        vf_parts.append(f"scale={width}:-1")

    cmd = [
        "ffmpeg",
        "-v", "quiet",
        "-nostdin",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", f"select=eq(n\\,{frame_number})" + (("," + ",".join(vf_parts)) if vf_parts else ""),
        "-frames:v", "1",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30, check=False
        )
        if result.returncode != 0:
            return b""
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return b""


def extract_frame_at_time(
    file_path: Path,
    time_seconds: float,
    stream_index: int = 0,
    width: int | None = None,
    pix_fmt: str = "gray16le",
) -> bytes:
    """Extract a frame at a specific timestamp.

    Uses seeking for performance on large files.

    Args:
        file_path: Path to the media file.
        time_seconds: Timestamp in seconds.
        stream_index: Video stream index.
        width: Target width for scaling (None = original).
        pix_fmt: Output pixel format.

    Returns:
        Raw pixel data as bytes.
    """
    vf_parts = []
    if width:
        vf_parts.append(f"scale={width}:-1")

    vf = ",".join(vf_parts) if vf_parts else None

    cmd = [
        "ffmpeg",
        "-v", "quiet",
        "-nostdin",
        "-ss", f"{time_seconds:.6f}",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-frames:v", "1",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30, check=False
        )
        if result.returncode != 0:
            return b""
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return b""


def generate_proxy(
    file_path: Path,
    output_path: Path,
    stream_index: int = 0,
    width: int = 480,
    fps: float | None = None,
) -> bool:
    """Generate a low-resolution grayscale proxy video.

    Args:
        file_path: Input video path.
        output_path: Output proxy path.
        stream_index: Video stream index to use.
        width: Target width.
        fps: Target framerate (None = same as source).

    Returns:
        True if successful.
    """
    vf_parts = [f"scale={width}:-1", "format=gray"]
    if fps:
        vf_parts.insert(0, f"fps={fps}")

    cmd = [
        "ffmpeg",
        "-v", "quiet",
        "-nostdin",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", ",".join(vf_parts),
        "-an",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=600, check=False
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
