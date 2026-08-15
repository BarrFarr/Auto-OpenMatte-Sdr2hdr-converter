"""Source inspection — parse ffprobe output into SourceInfo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from auto_openmatte.core.exceptions import InspectionError
from auto_openmatte.core.models import (
    ColorPrimaries,
    FrameRateType,
    HDRFormat,
    HDRMetadata,
    SourceInfo,
    SourceRole,
    TransferFunction,
    VideoStreamInfo,
)
from auto_openmatte.utils.ffmpeg import run_ffprobe, run_ffprobe_streams_only


def _parse_fps(stream: dict[str, Any]) -> tuple[float, str]:
    """Parse FPS from stream data.

    Returns:
        Tuple of (fps_float, fps_rational_string).
    """
    # Try r_frame_rate first (reliable for CFR)
    r_rate = stream.get("r_frame_rate", "0/1")
    avg_rate = stream.get("avg_frame_rate", "0/1")

    def _eval_rational(s: str) -> float:
        parts = s.split("/")
        if len(parts) == 2:
            num, den = int(parts[0]), int(parts[1])
            return num / den if den > 0 else 0.0
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    fps = _eval_rational(r_rate)
    if fps <= 0:
        fps = _eval_rational(avg_rate)

    rational_str = r_rate if fps > 0 else avg_rate
    return fps, rational_str


def _parse_transfer(stream: dict[str, Any]) -> TransferFunction:
    """Parse transfer characteristics from stream data."""
    transfer_str = stream.get("color_transfer", "unknown").lower()
    mapping = {
        "smpte2084": TransferFunction.PQ,
        "arib-std-b67": TransferFunction.HLG,
        "bt709": TransferFunction.BT709,
        "iec61966-2-1": TransferFunction.BT709,  # sRGB ≈ BT.709 for video
        "bt470bg": TransferFunction.GAMMA28,
        "smpte170m": TransferFunction.BT709,
    }
    return mapping.get(transfer_str, TransferFunction.UNKNOWN)


def _parse_primaries(stream: dict[str, Any]) -> ColorPrimaries:
    """Parse color primaries from stream data."""
    primaries_str = stream.get("color_primaries", "unknown").lower()
    mapping = {
        "bt709": ColorPrimaries.BT709,
        "bt2020": ColorPrimaries.BT2020,
        "smpte432": ColorPrimaries.DCI_P3,
    }
    return mapping.get(primaries_str, ColorPrimaries.UNKNOWN)


def _parse_bit_depth(stream: dict[str, Any]) -> int:
    """Parse bit depth from stream data."""
    # Try bits_per_raw_sample first
    bps = stream.get("bits_per_raw_sample")
    if bps:
        try:
            return int(bps)
        except (ValueError, TypeError):
            pass

    # Infer from pix_fmt
    pix_fmt = stream.get("pix_fmt", "")
    if "10" in pix_fmt:
        return 10
    elif "12" in pix_fmt:
        return 12
    elif "16" in pix_fmt:
        return 16
    return 8


def _parse_chroma(pix_fmt: str) -> str:
    """Infer chroma subsampling from pixel format name."""
    if "444" in pix_fmt:
        return "4:4:4"
    elif "422" in pix_fmt:
        return "4:2:2"
    elif "420" in pix_fmt:
        return "4:2:0"
    elif "gray" in pix_fmt:
        return "gray"
    return "unknown"


def _detect_vfr(stream: dict[str, Any]) -> FrameRateType:
    """Detect if stream is variable frame rate."""
    r_rate = stream.get("r_frame_rate", "0/1")
    avg_rate = stream.get("avg_frame_rate", "0/1")

    def _eval(s: str) -> float:
        parts = s.split("/")
        if len(parts) == 2:
            num, den = int(parts[0]), int(parts[1])
            return num / den if den > 0 else 0.0
        return float(s) if s else 0.0

    r = _eval(r_rate)
    avg = _eval(avg_rate)

    # If r_frame_rate and avg_frame_rate differ significantly, likely VFR
    if r > 0 and avg > 0 and abs(r - avg) / max(r, avg) > 0.01:
        return FrameRateType.VFR
    return FrameRateType.CFR


def _parse_stream(stream: dict[str, Any]) -> VideoStreamInfo:
    """Parse a single video stream dict into VideoStreamInfo."""
    fps, fps_rational = _parse_fps(stream)
    pix_fmt = stream.get("pix_fmt", "")

    # Duration
    duration = 0.0
    if "duration" in stream:
        try:
            duration = float(stream["duration"])
        except (ValueError, TypeError):
            pass

    # Frame count
    frame_count = None
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        try:
            frame_count = int(nb_frames)
        except (ValueError, TypeError):
            pass

    # Disposition
    disp = stream.get("disposition", {})
    is_default = disp.get("default", 0) == 1 if isinstance(disp, dict) else False

    return VideoStreamInfo(
        index=int(stream.get("index", 0)),
        codec=stream.get("codec_name", "unknown"),
        profile=stream.get("profile"),
        level=int(stream["level"]) if "level" in stream else None,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=fps,
        fps_rational=fps_rational,
        frame_rate_type=_detect_vfr(stream),
        duration_seconds=duration,
        frame_count=frame_count,
        pix_fmt=pix_fmt,
        bit_depth=_parse_bit_depth(stream),
        chroma_subsampling=_parse_chroma(pix_fmt),
        color_range=stream.get("color_range", "unknown"),
        color_primaries=_parse_primaries(stream),
        transfer=_parse_transfer(stream),
        matrix_coefficients=stream.get("color_space", "unknown"),
        is_default=is_default,
    )


def _detect_hdr_metadata(probe_data: dict[str, Any]) -> HDRMetadata:
    """Detect HDR metadata from ffprobe output (including frame side data)."""
    metadata = HDRMetadata()

    # Check stream-level side data
    streams = probe_data.get("streams", [])
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        side_data_list = stream.get("side_data_list", [])
        for sd in side_data_list:
            metadata.side_data.append(sd)
            sd_type = sd.get("side_data_type", "")

            if "Mastering display" in sd_type:
                # Parse mastering display metadata
                parts = []
                for key in ["red_x", "red_y", "green_x", "green_y", "blue_x", "blue_y",
                            "white_point_x", "white_point_y", "min_luminance", "max_luminance"]:
                    if key in sd:
                        parts.append(f"{key}={sd[key]}")
                if parts:
                    metadata.mastering_display = "; ".join(parts)

            if "Content light level" in sd_type:
                metadata.max_cll = sd.get("max_content")
                metadata.max_fall = sd.get("max_average")

    # Check frame-level side data
    frames = probe_data.get("frames", [])
    for frame in frames:
        side_data_list = frame.get("side_data_list", [])
        for sd in side_data_list:
            sd_type = sd.get("side_data_type", "")

            if "Mastering display" in sd_type and not metadata.mastering_display:
                parts = []
                for key in ["red_x", "red_y", "green_x", "green_y", "blue_x", "blue_y",
                            "white_point_x", "white_point_y", "min_luminance", "max_luminance"]:
                    if key in sd:
                        parts.append(f"{key}={sd[key]}")
                if parts:
                    metadata.mastering_display = "; ".join(parts)

            if "Content light level" in sd_type:
                if metadata.max_cll is None:
                    metadata.max_cll = sd.get("max_content")
                if metadata.max_fall is None:
                    metadata.max_fall = sd.get("max_average")

            if "Dolby Vision" in sd_type:
                metadata.format = HDRFormat.DOLBY_VISION

            if "HDR10+" in sd_type or "HDR Dynamic" in sd_type:
                metadata.format = HDRFormat.HDR10_PLUS

    return metadata


def _classify_hdr_format(
    stream: VideoStreamInfo, metadata: HDRMetadata
) -> HDRFormat:
    """Classify the HDR format based on stream info and metadata."""
    # Already detected DV or HDR10+ from side data
    if metadata.format in (HDRFormat.DOLBY_VISION, HDRFormat.HDR10_PLUS):
        return metadata.format

    # PQ transfer = HDR10 (or HDR10+ if dynamic metadata present)
    if stream.transfer == TransferFunction.PQ:
        if stream.color_primaries == ColorPrimaries.BT2020:
            return HDRFormat.HDR10
        # PQ without BT.2020 is unusual but still HDR
        return HDRFormat.HDR10

    # HLG transfer
    if stream.transfer == TransferFunction.HLG:
        return HDRFormat.HLG

    # SDR
    return HDRFormat.SDR


def inspect_source(file_path: Path) -> SourceInfo:
    """Inspect a media file and return complete source information.

    Args:
        file_path: Path to the media file.

    Returns:
        SourceInfo with all detected metadata.

    Raises:
        InspectionError: If the file cannot be inspected.
    """
    if not file_path.exists():
        raise InspectionError(f"File not found: {file_path}")

    # Run full ffprobe (with first frame for side data)
    probe_data = run_ffprobe(file_path)

    # Parse format info
    fmt = probe_data.get("format", {})
    container = fmt.get("format_name", "unknown")

    # Parse video streams
    streams_data = probe_data.get("streams", [])
    video_streams = []
    for s in streams_data:
        if s.get("codec_type") == "video":
            video_streams.append(_parse_stream(s))

    if not video_streams:
        raise InspectionError(f"No video streams found in {file_path}")

    # Try to get duration from format if not on stream
    format_duration = 0.0
    if "duration" in fmt:
        try:
            format_duration = float(fmt["duration"])
        except (ValueError, TypeError):
            pass

    for vs in video_streams:
        if vs.duration_seconds <= 0 and format_duration > 0:
            vs.duration_seconds = format_duration

    # Detect HDR metadata
    hdr_metadata = _detect_hdr_metadata(probe_data)

    # Build SourceInfo
    source = SourceInfo(
        path=file_path,
        container=container,
        video_streams=video_streams,
        hdr_metadata=hdr_metadata,
    )

    return source
