"""Time range specification and parsing utilities.

All user-facing ranges refer to the HDR timeline exclusively.
Open Matte ranges are derived automatically from the sync model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from auto_openmatte.core.exceptions import AutoOpenMatteError


class RangeError(AutoOpenMatteError):
    """Error in time range specification."""


def parse_time(time_str: str) -> float:
    """Parse a time string into seconds.

    Supported formats:
        HH:MM:SS.mmm
        HH:MM:SS
        MM:SS.mmm
        MM:SS
        SS.mmm
        SS (plain seconds as number)

    Args:
        time_str: Time string to parse.

    Returns:
        Time in seconds (float).

    Raises:
        RangeError: If the format is not recognized.
    """
    time_str = time_str.strip()

    # Try plain number first
    try:
        return float(time_str)
    except ValueError:
        pass

    # Try HH:MM:SS.mmm or HH:MM:SS
    pattern = r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$"
    match = re.match(pattern, time_str)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600.0 + minutes * 60.0 + seconds

    raise RangeError(
        f"Cannot parse time '{time_str}'. "
        "Expected formats: HH:MM:SS.mmm, HH:MM:SS, MM:SS, or seconds."
    )


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 0:
        return f"-{format_time(-seconds)}"
    hours = int(seconds // 3600)
    remainder = seconds - hours * 3600
    minutes = int(remainder // 60)
    secs = remainder - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class RangeSpec:
    """Specification of a processing range on the HDR timeline.

    All times/frames refer to the HDR source. The corresponding
    Open Matte range is computed from the sync model's frame_offset.
    """

    # HDR time range (seconds)
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    # HDR frame range (computed from time + FPS)
    start_frame: int = 0
    end_frame: int = 0

    # Context window (seconds, NOT included in final output)
    context_seconds: float = 2.0

    # Effective analysis range including context
    analysis_start_seconds: float = 0.0
    analysis_end_seconds: float = 0.0
    analysis_start_frame: int = 0
    analysis_end_frame: int = 0

    # Mapped Open Matte range (from sync model)
    om_start_frame: int = 0
    om_end_frame: int = 0
    om_analysis_start_frame: int = 0
    om_analysis_end_frame: int = 0

    # Whether this is a full-source range (no user selection)
    is_full_range: bool = True

    @property
    def duration_seconds(self) -> float:
        """Duration of the requested range (excluding context)."""
        return self.end_seconds - self.start_seconds

    @property
    def duration_frames(self) -> int:
        """Duration in frames (excluding context)."""
        return self.end_frame - self.start_frame

    @property
    def analysis_duration_seconds(self) -> float:
        """Duration including context."""
        return self.analysis_end_seconds - self.analysis_start_seconds

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for project.json."""
        return {
            "timeline": "HDR",
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "context_seconds": self.context_seconds,
            "analysis_start_seconds": self.analysis_start_seconds,
            "analysis_end_seconds": self.analysis_end_seconds,
            "analysis_start_frame": self.analysis_start_frame,
            "analysis_end_frame": self.analysis_end_frame,
            "om_start_frame": self.om_start_frame,
            "om_end_frame": self.om_end_frame,
            "om_analysis_start_frame": self.om_analysis_start_frame,
            "om_analysis_end_frame": self.om_analysis_end_frame,
            "is_full_range": self.is_full_range,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RangeSpec:
        """Deserialize from dict."""
        return cls(
            start_seconds=data.get("start_seconds", 0.0),
            end_seconds=data.get("end_seconds", 0.0),
            start_frame=data.get("start_frame", 0),
            end_frame=data.get("end_frame", 0),
            context_seconds=data.get("context_seconds", 2.0),
            analysis_start_seconds=data.get("analysis_start_seconds", 0.0),
            analysis_end_seconds=data.get("analysis_end_seconds", 0.0),
            analysis_start_frame=data.get("analysis_start_frame", 0),
            analysis_end_frame=data.get("analysis_end_frame", 0),
            om_start_frame=data.get("om_start_frame", 0),
            om_end_frame=data.get("om_end_frame", 0),
            om_analysis_start_frame=data.get("om_analysis_start_frame", 0),
            om_analysis_end_frame=data.get("om_analysis_end_frame", 0),
            is_full_range=data.get("is_full_range", True),
        )


def build_range_spec(
    start: float | None = None,
    end: float | None = None,
    duration: float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    context: float = 2.0,
    fps: float = 24.0,
    total_frames: int = 0,
    total_duration: float = 0.0,
    frame_offset: int = 0,
) -> RangeSpec:
    """Build a RangeSpec from user-provided arguments.

    All inputs refer to the HDR timeline. The OM range is computed
    from frame_offset (OM_frame = HDR_frame + frame_offset).

    Args:
        start: Start time in seconds (HDR timeline).
        end: End time in seconds (HDR timeline).
        duration: Duration in seconds (alternative to end).
        start_frame: Start frame number (alternative to start time).
        end_frame: End frame number (alternative to end time).
        context: Context window in seconds.
        fps: HDR source FPS.
        total_frames: Total HDR frame count.
        total_duration: Total HDR duration in seconds.
        frame_offset: Sync model frame offset (OM = HDR + offset).

    Returns:
        Fully populated RangeSpec.

    Raises:
        RangeError: If arguments are inconsistent or invalid.
    """
    # Determine if this is a full-range request
    is_full = (
        start is None
        and end is None
        and duration is None
        and start_frame is None
        and end_frame is None
    )

    if is_full:
        # Full source processing
        spec = RangeSpec(
            start_seconds=0.0,
            end_seconds=total_duration,
            start_frame=0,
            end_frame=total_frames,
            context_seconds=0.0,
            analysis_start_seconds=0.0,
            analysis_end_seconds=total_duration,
            analysis_start_frame=0,
            analysis_end_frame=total_frames,
            om_start_frame=frame_offset,
            om_end_frame=total_frames + frame_offset,
            om_analysis_start_frame=frame_offset,
            om_analysis_end_frame=total_frames + frame_offset,
            is_full_range=True,
        )
        return spec

    # --- Resolve time-based range ---
    if start is not None and end is not None and duration is not None:
        # All three specified — check consistency
        expected_end = start + duration
        if abs(expected_end - end) > 0.01:
            raise RangeError(
                f"Inconsistent range: --start {start} + --duration {duration} = "
                f"{expected_end}, but --end {end} was specified."
            )

    if start is None and start_frame is not None:
        start = start_frame / fps
    elif start is None:
        start = 0.0

    if end is None and duration is not None:
        end = start + duration
    elif end is None and end_frame is not None:
        end = end_frame / fps
    elif end is None:
        end = total_duration

    # Validate
    if end <= start:
        raise RangeError(f"End time ({end:.3f}s) must be after start ({start:.3f}s).")
    if start < 0:
        raise RangeError(f"Start time ({start:.3f}s) cannot be negative.")
    if end > total_duration and total_duration > 0:
        raise RangeError(
            f"End time ({end:.3f}s) exceeds source duration ({total_duration:.3f}s)."
        )

    # --- Frame-based validation ---
    s_frame = int(round(start * fps))
    e_frame = int(round(end * fps))

    if start_frame is not None and abs(start_frame - s_frame) > 1:
        raise RangeError(
            f"Inconsistent: --start-frame {start_frame} vs "
            f"time-derived frame {s_frame}."
        )
    if end_frame is not None and abs(end_frame - e_frame) > 1:
        raise RangeError(
            f"Inconsistent: --end-frame {end_frame} vs "
            f"time-derived frame {e_frame}."
        )

    if start_frame is not None:
        s_frame = start_frame
    if end_frame is not None:
        e_frame = end_frame

    # --- Context window ---
    analysis_start = max(0.0, start - context)
    analysis_end = min(total_duration, end + context) if total_duration > 0 else end + context
    analysis_s_frame = max(0, int(round(analysis_start * fps)))
    analysis_e_frame = (
        min(total_frames, int(round(analysis_end * fps))) if total_frames > 0
        else int(round(analysis_end * fps))
    )

    # --- Map to Open Matte ---
    om_start = s_frame + frame_offset
    om_end = e_frame + frame_offset
    om_analysis_start = analysis_s_frame + frame_offset
    om_analysis_end = analysis_e_frame + frame_offset

    spec = RangeSpec(
        start_seconds=start,
        end_seconds=end,
        start_frame=s_frame,
        end_frame=e_frame,
        context_seconds=context,
        analysis_start_seconds=analysis_start,
        analysis_end_seconds=analysis_end,
        analysis_start_frame=analysis_s_frame,
        analysis_end_frame=analysis_e_frame,
        om_start_frame=om_start,
        om_end_frame=om_end,
        om_analysis_start_frame=om_analysis_start,
        om_analysis_end_frame=om_analysis_end,
        is_full_range=False,
    )
    return spec
