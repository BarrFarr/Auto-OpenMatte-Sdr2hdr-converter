"""Video stream selection for multi-stream containers.

Selects the best video stream based on quality indicators.
"""

from __future__ import annotations

from auto_openmatte.core.exceptions import StreamSelectionError
from auto_openmatte.core.models import SourceInfo, TransferFunction, VideoStreamInfo


def _stream_quality_score(stream: VideoStreamInfo) -> int:
    """Compute a quality score for a video stream.

    Higher score = better quality / more suitable for processing.
    """
    score = 0

    # Resolution (primary factor)
    pixels = stream.width * stream.height
    score += pixels // 10000  # ~830 for 4K UHD

    # Bit depth
    score += stream.bit_depth * 5  # 50 for 10-bit, 40 for 8-bit

    # HDR transfer function indicates a high-quality master
    if stream.transfer in (TransferFunction.PQ, TransferFunction.HLG):
        score += 200

    # Default disposition
    if stream.is_default:
        score += 50

    # Penalize very low resolution (probably a thumbnail/proxy)
    if pixels < 100000:  # Less than ~316x316
        score -= 1000

    return score


def select_video_stream(source: SourceInfo) -> SourceInfo:
    """Select the best video stream from the source.

    Modifies source.selected_stream in place and returns the source.

    Args:
        source: SourceInfo with video_streams populated.

    Returns:
        The same SourceInfo with selected_stream set.

    Raises:
        StreamSelectionError: If no suitable stream can be found.
    """
    if not source.video_streams:
        raise StreamSelectionError(f"No video streams in {source.path}")

    # Single stream — trivial
    if len(source.video_streams) == 1:
        source.selected_stream = source.video_streams[0]
        return source

    # Multiple streams — score and select best
    scored = [(s, _stream_quality_score(s)) for s in source.video_streams]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_stream, best_score = scored[0]

    # Sanity check: best stream should be reasonable
    if best_stream.width < 100 or best_stream.height < 100:
        raise StreamSelectionError(
            f"Best stream in {source.path} has implausible resolution: "
            f"{best_stream.width}x{best_stream.height}"
        )

    source.selected_stream = best_stream
    return source
