"""Source inspection via ffprobe - metadata extraction and parsing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from auto_openmatte.core.exceptions import InvalidSourceError
from auto_openmatte.core.models import SourceInfo


def run_ffprobe(filepath: str) -> dict[str, Any]:
    """Execute ffprobe on a video file and return parsed JSON output.

    Args:
        filepath: Path to the video file.

    Returns:
        Parsed dict from ffprobe JSON output.

    Raises:
        InvalidSourceError: If ffprobe fails, file not found, or output
            cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise InvalidSourceError(f"File not found: {filepath}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        raise InvalidSourceError(
            "ffprobe not found. Please install FFmpeg to use this tool."
        )
    except subprocess.TimeoutExpired:
        raise InvalidSourceError(
            f"ffprobe timed out processing: {filepath}"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "Unknown error"
        raise InvalidSourceError(
            f"ffprobe failed for '{filepath}': {stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise InvalidSourceError(
            f"Failed to parse ffprobe JSON output for '{filepath}': {e}"
        )

    if "streams" not in data or not data["streams"]:
        raise InvalidSourceError(
            f"No video streams found in '{filepath}'"
        )

    return data


def _parse_fps(stream: dict[str, Any]) -> float:
    """Parse frame rate from stream data, trying multiple fields."""
    # Try r_frame_rate first (real frame rate)
    r_frame_rate = stream.get("r_frame_rate", "")
    if r_frame_rate and "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f > 0:
                return num_f / den_f
        except (ValueError, ZeroDivisionError):
            pass

    # Try avg_frame_rate
    avg_frame_rate = stream.get("avg_frame_rate", "")
    if avg_frame_rate and "/" in avg_frame_rate:
        num, den = avg_frame_rate.split("/")
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f > 0:
                return num_f / den_f
        except (ValueError, ZeroDivisionError):
            pass

    return 0.0


def _parse_bit_depth(stream: dict[str, Any]) -> int:
    """Extract bit depth from stream data."""
    # Direct bits_per_raw_sample
    bits = stream.get("bits_per_raw_sample")
    if bits:
        try:
            return int(bits)
        except (ValueError, TypeError):
            pass

    # From pixel format name (e.g., yuv420p10le -> 10)
    pix_fmt = stream.get("pix_fmt", "")
    if "10" in pix_fmt:
        return 10
    if "12" in pix_fmt:
        return 12
    if "16" in pix_fmt:
        return 16

    return 8


def _parse_chroma_subsampling(stream: dict[str, Any]) -> str:
    """Infer chroma subsampling from pixel format."""
    pix_fmt = stream.get("pix_fmt", "")
    if "444" in pix_fmt:
        return "4:4:4"
    if "422" in pix_fmt:
        return "4:2:2"
    if "420" in pix_fmt:
        return "4:2:0"
    # Check if it contains common format names
    if pix_fmt.startswith("yuv420") or pix_fmt.startswith("p010"):
        return "4:2:0"
    if pix_fmt.startswith("yuv422"):
        return "4:2:2"
    if pix_fmt.startswith("yuv444"):
        return "4:4:4"
    return "unknown"


def _extract_side_data(stream: dict[str, Any]) -> dict[str, str]:
    """Extract HDR side_data entries from stream."""
    result: dict[str, str] = {}
    side_data_list = stream.get("side_data_list", [])

    for entry in side_data_list:
        side_type = entry.get("side_data_type", "")

        if "Mastering display" in side_type:
            # Build mastering display string from components
            parts = []
            for key in ("red_x", "red_y", "green_x", "green_y",
                        "blue_x", "blue_y", "white_point_x", "white_point_y",
                        "min_luminance", "max_luminance"):
                if key in entry:
                    parts.append(f"{key}={entry[key]}")
            result["mastering_display"] = ", ".join(parts) if parts else str(entry)

        elif "Content light level" in side_type:
            if "max_content" in entry:
                result["max_cll"] = str(entry["max_content"])
            if "max_average" in entry:
                result["max_fall"] = str(entry["max_average"])

        elif "Dolby Vision" in side_type:
            result["dolby_vision"] = "true"

        elif "HDR10+" in side_type or "HDR Dynamic" in side_type:
            result["hdr10plus"] = "true"

    return result


def _detect_vfr_cfr(stream: dict[str, Any]) -> str:
    """Detect whether stream is VFR or CFR."""
    r_frame_rate = stream.get("r_frame_rate", "")
    avg_frame_rate = stream.get("avg_frame_rate", "")

    if r_frame_rate and avg_frame_rate and r_frame_rate != avg_frame_rate:
        # Different real and average frame rates may indicate VFR
        return "VFR"
    return "CFR"


def extract_source_info(ffprobe_data: dict[str, Any], filepath: str = "") -> SourceInfo:
    """Parse ffprobe JSON output into a SourceInfo dataclass.

    Args:
        ffprobe_data: Parsed JSON dict from ffprobe output.
        filepath: Original file path (for reference in SourceInfo).

    Returns:
        SourceInfo dataclass with all metadata fields populated.
    """
    streams = ffprobe_data.get("streams", [])
    if not streams:
        raise InvalidSourceError("No video streams in ffprobe data")

    # Use first video stream by default
    stream = streams[0]
    fmt = ffprobe_data.get("format", {})

    # Basic video properties
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    fps = _parse_fps(stream)
    duration = float(stream.get("duration", 0) or fmt.get("duration", 0) or 0)
    frame_count_raw = stream.get("nb_frames", "0")
    try:
        frame_count = int(frame_count_raw)
    except (ValueError, TypeError):
        # Estimate from duration and fps
        frame_count = int(duration * fps) if fps > 0 else 0

    # Codec info
    codec = stream.get("codec_name", "unknown")
    profile = stream.get("profile", "unknown")
    level_raw = stream.get("level", "")
    level = str(level_raw) if level_raw else "unknown"

    # Container format
    container = fmt.get("format_name", "unknown")

    # Pixel format and derived values
    pixel_format = stream.get("pix_fmt", "unknown")
    bit_depth = _parse_bit_depth(stream)
    chroma_subsampling = _parse_chroma_subsampling(stream)

    # Color metadata
    color_range = stream.get("color_range", "unknown")
    color_primaries = stream.get("color_primaries", "unknown")
    transfer_characteristics = stream.get("color_transfer", "unknown")
    matrix_coefficients = stream.get("color_space", "unknown")

    # Frame rate info
    r_frame_rate = stream.get("r_frame_rate", "")
    nominal_fps = r_frame_rate if r_frame_rate else str(fps)
    vfr_cfr = _detect_vfr_cfr(stream)

    # HDR side data
    side_data = _extract_side_data(stream)
    mastering_display = side_data.get("mastering_display", "")
    max_cll = int(side_data.get("max_cll", "0") or "0")
    max_fall = int(side_data.get("max_fall", "0") or "0")

    return SourceInfo(
        path=filepath,
        container=container,
        codec=codec,
        profile=profile,
        level=level,
        width=width,
        height=height,
        fps=fps,
        nominal_fps=nominal_fps,
        vfr_cfr=vfr_cfr,
        duration=duration,
        frame_count=frame_count,
        pixel_format=pixel_format,
        bit_depth=bit_depth,
        chroma_subsampling=chroma_subsampling,
        color_range=color_range,
        color_primaries=color_primaries,
        transfer_characteristics=transfer_characteristics,
        matrix_coefficients=matrix_coefficients,
        mastering_display=mastering_display,
        max_cll=max_cll,
        max_fall=max_fall,
        color_space=matrix_coefficients,
        color_transfer=transfer_characteristics,
    )


def inspect_source(filepath: str) -> SourceInfo:
    """Convenience function: run ffprobe and extract source info.

    Args:
        filepath: Path to the video file.

    Returns:
        SourceInfo with all metadata fields populated.

    Raises:
        InvalidSourceError: If ffprobe fails or file is invalid.
    """
    data = run_ffprobe(filepath)
    return extract_source_info(data, filepath)
