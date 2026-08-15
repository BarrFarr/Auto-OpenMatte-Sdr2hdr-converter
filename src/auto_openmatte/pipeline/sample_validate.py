"""Output validation for generated sample clips.

Verifies that the output meets all requirements:
- File exists
- Duration matches requested range (not including context)
- FPS matches HDR source
- Frame count is correct
- Resolution is correct
- HDR metadata is present
- No audio stream
- No context frames leaked into output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auto_openmatte.core.range_spec import RangeSpec
from auto_openmatte.utils.ffmpeg import run_ffprobe_streams_only

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation."""

    passed: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        """Add a validation check result."""
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False
            self.errors.append(f"{name}: {detail}")

    def add_warning(self, msg: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(msg)


def validate_sample_output(
    output_path: Path,
    range_spec: RangeSpec,
    expected_fps: float,
    expected_width: int,
    expected_height: int,
    expect_hdr: bool = True,
) -> ValidationResult:
    """Validate a generated sample clip against the requested range.

    Args:
        output_path: Path to the output video.
        range_spec: The range that was requested.
        expected_fps: Expected FPS (from HDR source).
        expected_width: Expected output width.
        expected_height: Expected output height.
        expect_hdr: Whether HDR metadata should be present.

    Returns:
        ValidationResult with all check results.
    """
    result = ValidationResult()

    # Check 1: File exists
    if not output_path.exists():
        result.add_check("file_exists", False, f"Output not found: {output_path}")
        return result  # Can't continue without a file
    result.add_check("file_exists", True, str(output_path))

    # Check 2: File is not empty
    file_size = output_path.stat().st_size
    if file_size == 0:
        result.add_check("file_not_empty", False, "Output file is 0 bytes")
        return result
    result.add_check("file_not_empty", True, f"{file_size} bytes")

    # Probe the output
    try:
        probe = run_ffprobe_streams_only(output_path)
    except Exception as e:
        result.add_check("probe_readable", False, f"ffprobe failed: {e}")
        return result
    result.add_check("probe_readable", True)

    streams = probe.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    # Check 3: Has video stream
    if not video_streams:
        result.add_check("has_video", False, "No video stream in output")
        return result
    result.add_check("has_video", True)

    vs = video_streams[0]

    # Check 4: No audio (audio excluded from project)
    if audio_streams:
        result.add_check("no_audio", False, f"{len(audio_streams)} audio stream(s) found")
    else:
        result.add_check("no_audio", True)

    # Check 5: Resolution
    out_w = int(vs.get("width", 0))
    out_h = int(vs.get("height", 0))
    if out_w == expected_width and out_h == expected_height:
        result.add_check("resolution", True, f"{out_w}x{out_h}")
    else:
        result.add_check(
            "resolution", False,
            f"Got {out_w}x{out_h}, expected {expected_width}x{expected_height}"
        )

    # Check 6: FPS
    r_rate = vs.get("r_frame_rate", "0/1")
    try:
        parts = r_rate.split("/")
        out_fps = int(parts[0]) / int(parts[1]) if len(parts) == 2 else float(r_rate)
    except (ValueError, ZeroDivisionError):
        out_fps = 0.0

    fps_tolerance = 0.01
    if abs(out_fps - expected_fps) <= fps_tolerance:
        result.add_check("fps", True, f"{out_fps:.3f}")
    else:
        result.add_check(
            "fps", False,
            f"Got {out_fps:.3f}, expected {expected_fps:.3f}"
        )

    # Check 7: Duration matches requested range (not context)
    expected_duration = range_spec.duration_seconds
    fmt_data = probe.get("format", {})
    out_duration = float(fmt_data.get("duration", 0))
    if out_duration == 0:
        out_duration = float(vs.get("duration", 0))

    # Allow 1-frame tolerance
    frame_duration = 1.0 / expected_fps if expected_fps > 0 else 0.042
    duration_tolerance = frame_duration * 2  # 2 frames tolerance

    if abs(out_duration - expected_duration) <= duration_tolerance:
        result.add_check(
            "duration", True,
            f"{out_duration:.3f}s (expected {expected_duration:.3f}s)"
        )
    else:
        result.add_check(
            "duration", False,
            f"Got {out_duration:.3f}s, expected {expected_duration:.3f}s "
            f"(tolerance ±{duration_tolerance:.3f}s)"
        )

    # Check 8: Frame count
    expected_frames = range_spec.duration_frames
    out_frames = vs.get("nb_frames")
    if out_frames and out_frames != "N/A":
        out_frames_int = int(out_frames)
        frame_tolerance = 2
        if abs(out_frames_int - expected_frames) <= frame_tolerance:
            result.add_check(
                "frame_count", True,
                f"{out_frames_int} (expected {expected_frames})"
            )
        else:
            result.add_check(
                "frame_count", False,
                f"Got {out_frames_int}, expected {expected_frames} (±{frame_tolerance})"
            )
    else:
        result.add_warning("Frame count not available in output metadata")

    # Check 9: HDR metadata (if expected)
    if expect_hdr:
        transfer = vs.get("color_transfer", "")
        primaries = vs.get("color_primaries", "")
        has_pq = transfer in ("smpte2084", "arib-std-b67")
        has_bt2020 = primaries == "bt2020"

        if has_pq and has_bt2020:
            result.add_check(
                "hdr_metadata", True,
                f"transfer={transfer}, primaries={primaries}"
            )
        elif has_pq:
            result.add_check("hdr_metadata", True, f"transfer={transfer}")
            result.add_warning(f"Primaries are '{primaries}', expected bt2020")
        else:
            result.add_check(
                "hdr_metadata", False,
                f"transfer={transfer}, primaries={primaries} — expected PQ/HLG + BT.2020"
            )

    # Summary logging
    passed_count = sum(1 for c in result.checks if c["passed"])
    total_count = len(result.checks)
    if result.passed:
        logger.info(f"Validation PASSED: {passed_count}/{total_count} checks OK")
    else:
        logger.error(
            f"Validation FAILED: {passed_count}/{total_count} checks passed"
        )
        for err in result.errors:
            logger.error(f"  ✗ {err}")

    for warn in result.warnings:
        logger.warning(f"  ⚠ {warn}")

    return result
