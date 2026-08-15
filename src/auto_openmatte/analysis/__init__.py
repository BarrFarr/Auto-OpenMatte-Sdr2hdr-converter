"""Analysis modules: source inspection, synchronization, and shot detection."""

from auto_openmatte.analysis.hdr_detect import (
    detect_hdr_standard,
    identify_roles,
    validate_hdr_support,
)
from auto_openmatte.analysis.inspect import (
    extract_source_info,
    inspect_source,
    run_ffprobe,
)
from auto_openmatte.analysis.stream_select import (
    list_video_streams,
    select_best_video_stream,
)

__all__ = [
    "detect_hdr_standard",
    "extract_source_info",
    "identify_roles",
    "inspect_source",
    "list_video_streams",
    "run_ffprobe",
    "select_best_video_stream",
    "validate_hdr_support",
]
