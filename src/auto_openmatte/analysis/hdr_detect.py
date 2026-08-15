"""HDR detection and source role assignment.

Determines which source is HDR and which is Open Matte.
Refuses to guess if ambiguous.
"""

from __future__ import annotations

from auto_openmatte.core.exceptions import HDRDetectionError, UnsupportedHDRError
from auto_openmatte.core.models import (
    HDRFormat,
    SourceInfo,
    SourceRole,
    TransferFunction,
    VideoStreamInfo,
)


def _hdr_score(source: SourceInfo) -> int:
    """Compute an HDR confidence score for a source.

    Higher score = more likely to be HDR.
    Returns 0 for clear SDR, high values for clear HDR.
    """
    score = 0
    stream = source.selected_stream
    if not stream:
        return 0

    # Transfer function is the primary indicator
    if stream.transfer == TransferFunction.PQ:
        score += 100
    elif stream.transfer == TransferFunction.HLG:
        score += 90

    # Color primaries
    if stream.color_primaries.value == "bt2020":
        score += 30

    # Bit depth
    if stream.bit_depth >= 10:
        score += 10

    # HDR metadata presence
    if source.hdr_metadata.mastering_display:
        score += 20
    if source.hdr_metadata.max_cll is not None:
        score += 10
    if source.hdr_metadata.format != HDRFormat.SDR:
        score += 50

    return score


def assign_roles(source_a: SourceInfo, source_b: SourceInfo) -> tuple[SourceInfo, SourceInfo]:
    """Determine which source is HDR reference and which is Open Matte.

    Args:
        source_a: First source (as provided by user, may be in any order).
        source_b: Second source.

    Returns:
        Tuple of (hdr_source, openmatte_source) with roles assigned.

    Raises:
        HDRDetectionError: If roles cannot be determined unambiguously.
        UnsupportedHDRError: If HDR format is detected but not supported.
    """
    score_a = _hdr_score(source_a)
    score_b = _hdr_score(source_b)

    # Check for unsupported HDR formats
    for source, label in [(source_a, "Source A"), (source_b, "Source B")]:
        if source.hdr_metadata.format == HDRFormat.DOLBY_VISION:
            raise UnsupportedHDRError(
                f"{label} ({source.path.name}) contains Dolby Vision. "
                "Detected HDR format is not supported by the current processing pipeline."
            )

    # Need clear separation
    if score_a == score_b:
        raise HDRDetectionError(
            f"Cannot determine HDR roles. Both sources have equal HDR score ({score_a}).\n"
            f"  Source A: {source_a.path.name} — "
            f"transfer={source_a.selected_stream.transfer.value if source_a.selected_stream else 'N/A'}\n"
            f"  Source B: {source_b.path.name} — "
            f"transfer={source_b.selected_stream.transfer.value if source_b.selected_stream else 'N/A'}\n"
            "Please verify your input files."
        )

    if score_a == 0 and score_b == 0:
        raise HDRDetectionError(
            "Neither source appears to be HDR.\n"
            f"  Source A: {source_a.path.name}\n"
            f"  Source B: {source_b.path.name}\n"
            "At least one source must be HDR (PQ or HLG transfer)."
        )

    # Minimum threshold for HDR identification
    max_score = max(score_a, score_b)
    min_score = min(score_a, score_b)

    if max_score < 50:
        raise HDRDetectionError(
            "No source has confident HDR indicators (score < 50).\n"
            f"  Source A: {source_a.path.name} — HDR score = {score_a}\n"
            f"  Source B: {source_b.path.name} — HDR score = {score_b}\n"
            "Expected: PQ/HLG transfer function with BT.2020 primaries."
        )

    # The source with the highest score difference is HDR
    if abs(score_a - score_b) < 20:
        raise HDRDetectionError(
            "Ambiguous HDR detection — scores too close.\n"
            f"  Source A: {source_a.path.name} — HDR score = {score_a}\n"
            f"  Source B: {source_b.path.name} — HDR score = {score_b}\n"
            "Cannot reliably determine roles. Please check your sources."
        )

    if score_a > score_b:
        hdr_source = source_a
        om_source = source_b
    else:
        hdr_source = source_b
        om_source = source_a

    # Assign roles
    hdr_source.role = SourceRole.HDR_REFERENCE
    om_source.role = SourceRole.OPEN_MATTE

    # Classify HDR format on the HDR source
    if hdr_source.selected_stream:
        from auto_openmatte.analysis.inspect import _classify_hdr_format
        hdr_source.hdr_metadata.format = _classify_hdr_format(
            hdr_source.selected_stream, hdr_source.hdr_metadata
        )

    # Verify the HDR format is supported
    supported = {HDRFormat.HDR10, HDRFormat.HDR10_PLUS, HDRFormat.HLG}
    if hdr_source.hdr_metadata.format not in supported:
        if hdr_source.hdr_metadata.format == HDRFormat.SDR:
            raise HDRDetectionError(
                "HDR source classified as SDR after detailed analysis. Check metadata."
            )
        raise UnsupportedHDRError(
            f"Detected HDR format '{hdr_source.hdr_metadata.format.value}' "
            "is not supported by the current processing pipeline."
        )

    return hdr_source, om_source
