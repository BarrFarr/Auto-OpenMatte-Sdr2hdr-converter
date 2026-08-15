"""Video stream selection for multi-stream containers."""

from __future__ import annotations

from typing import Any


def select_best_video_stream(streams_list: list[dict[str, Any]]) -> int:
    """Select the best video stream from a list of stream entries.

    Selection criteria (in priority order):
    1. Highest resolution (width * height)
    2. HDR metadata presence (side_data_list with mastering display)
    3. Highest bit depth
    4. Default disposition

    Args:
        streams_list: List of stream dicts from ffprobe JSON output
            (already filtered to video streams).

    Returns:
        Index into streams_list of the best stream.
    """
    if not streams_list:
        return 0

    def _score_stream(stream: dict[str, Any]) -> tuple[int, int, int, int]:
        """Score a stream for quality ranking. Higher is better."""
        # Resolution score
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        resolution = width * height

        # HDR metadata presence score
        has_hdr = 0
        side_data_list = stream.get("side_data_list", [])
        for entry in side_data_list:
            side_type = entry.get("side_data_type", "")
            if "Mastering display" in side_type or "Content light" in side_type:
                has_hdr = 1
                break

        # Bit depth score
        bit_depth = 8
        bits_raw = stream.get("bits_per_raw_sample")
        if bits_raw:
            try:
                bit_depth = int(bits_raw)
            except (ValueError, TypeError):
                pass
        else:
            pix_fmt = stream.get("pix_fmt", "")
            if "10" in pix_fmt:
                bit_depth = 10
            elif "12" in pix_fmt:
                bit_depth = 12

        # Default disposition score
        disposition = stream.get("disposition", {})
        is_default = int(disposition.get("default", 0)) if isinstance(disposition, dict) else 0

        return (resolution, has_hdr, bit_depth, is_default)

    best_idx = 0
    best_score = _score_stream(streams_list[0])

    for idx in range(1, len(streams_list)):
        score = _score_stream(streams_list[idx])
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def list_video_streams(streams_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a summary list of all video streams for display.

    Args:
        streams_list: List of stream dicts from ffprobe JSON output
            (already filtered to video streams).

    Returns:
        List of summary dicts with key properties of each stream.
    """
    summaries: list[dict[str, Any]] = []

    for idx, stream in enumerate(streams_list):
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        codec = stream.get("codec_name", "unknown")
        pix_fmt = stream.get("pix_fmt", "unknown")
        color_transfer = stream.get("color_transfer", "unknown")
        color_primaries = stream.get("color_primaries", "unknown")

        # Bit depth
        bit_depth = 8
        bits_raw = stream.get("bits_per_raw_sample")
        if bits_raw:
            try:
                bit_depth = int(bits_raw)
            except (ValueError, TypeError):
                pass
        else:
            if "10" in pix_fmt:
                bit_depth = 10
            elif "12" in pix_fmt:
                bit_depth = 12

        # HDR info
        has_hdr_metadata = False
        side_data_list = stream.get("side_data_list", [])
        for entry in side_data_list:
            side_type = entry.get("side_data_type", "")
            if "Mastering display" in side_type:
                has_hdr_metadata = True
                break

        # Disposition
        disposition = stream.get("disposition", {})
        is_default = bool(disposition.get("default", 0)) if isinstance(disposition, dict) else False

        summaries.append({
            "index": idx,
            "stream_index": stream.get("index", idx),
            "codec": codec,
            "width": width,
            "height": height,
            "pixel_format": pix_fmt,
            "bit_depth": bit_depth,
            "color_transfer": color_transfer,
            "color_primaries": color_primaries,
            "has_hdr_metadata": has_hdr_metadata,
            "is_default": is_default,
            "resolution": f"{width}x{height}",
        })

    return summaries
