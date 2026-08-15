"""Tests for time range specification and parsing."""

from __future__ import annotations

import pytest

from auto_openmatte.core.range_spec import (
    RangeError,
    RangeSpec,
    build_range_spec,
    format_time,
    parse_time,
)


class TestParseTime:
    """Tests for time string parsing."""

    def test_plain_seconds(self) -> None:
        assert parse_time("123.456") == pytest.approx(123.456)

    def test_plain_integer(self) -> None:
        assert parse_time("30") == pytest.approx(30.0)

    def test_mm_ss(self) -> None:
        assert parse_time("01:30") == pytest.approx(90.0)

    def test_mm_ss_millis(self) -> None:
        assert parse_time("01:30.500") == pytest.approx(90.5)

    def test_hh_mm_ss(self) -> None:
        assert parse_time("01:23:10") == pytest.approx(4990.0)

    def test_hh_mm_ss_millis(self) -> None:
        assert parse_time("01:23:10.500") == pytest.approx(4990.5)

    def test_zero(self) -> None:
        assert parse_time("00:00:00") == pytest.approx(0.0)

    def test_large_hours(self) -> None:
        assert parse_time("02:00:00") == pytest.approx(7200.0)

    def test_whitespace_stripped(self) -> None:
        assert parse_time("  01:23:10  ") == pytest.approx(4990.0)

    def test_invalid_raises(self) -> None:
        with pytest.raises(RangeError, match="Cannot parse time"):
            parse_time("not_a_time")

    def test_empty_raises(self) -> None:
        with pytest.raises(RangeError):
            parse_time("")


class TestFormatTime:
    """Tests for time formatting."""

    def test_zero(self) -> None:
        assert format_time(0.0) == "00:00:00.000"

    def test_seconds(self) -> None:
        assert format_time(30.5) == "00:00:30.500"

    def test_minutes(self) -> None:
        assert format_time(90.0) == "00:01:30.000"

    def test_hours(self) -> None:
        assert format_time(4990.0) == "01:23:10.000"

    def test_round_trip(self) -> None:
        """parse(format(x)) should return approximately x."""
        original = 4990.5
        formatted = format_time(original)
        parsed = parse_time(formatted)
        assert parsed == pytest.approx(original, abs=0.001)


class TestBuildRangeSpec:
    """Tests for building a RangeSpec from arguments."""

    def test_full_range_default(self) -> None:
        """No arguments → full source range."""
        spec = build_range_spec(
            fps=24.0, total_frames=172800, total_duration=7200.0, frame_offset=58
        )
        assert spec.is_full_range is True
        assert spec.start_seconds == 0.0
        assert spec.end_seconds == 7200.0
        assert spec.start_frame == 0
        assert spec.end_frame == 172800
        assert spec.om_start_frame == 58
        assert spec.om_end_frame == 172858

    def test_start_and_duration(self) -> None:
        """--start + --duration → correct range."""
        spec = build_range_spec(
            start=4990.0, duration=30.0,
            fps=24.0, total_frames=172800, total_duration=7200.0,
            frame_offset=58,
        )
        assert spec.is_full_range is False
        assert spec.start_seconds == pytest.approx(4990.0)
        assert spec.end_seconds == pytest.approx(5020.0)
        assert spec.duration_seconds == pytest.approx(30.0)
        assert spec.start_frame == 119760  # 4990 * 24
        assert spec.end_frame == 120480  # 5020 * 24

    def test_om_range_from_offset(self) -> None:
        """OM range should be HDR range + frame_offset."""
        spec = build_range_spec(
            start=100.0, end=110.0,
            fps=24.0, total_frames=172800, total_duration=7200.0,
            frame_offset=58,
        )
        assert spec.om_start_frame == spec.start_frame + 58
        assert spec.om_end_frame == spec.end_frame + 58

    def test_context_window(self) -> None:
        """Context window should expand analysis range but not output range."""
        spec = build_range_spec(
            start=100.0, end=110.0, context=2.0,
            fps=24.0, total_frames=172800, total_duration=7200.0,
            frame_offset=58,
        )
        # Output range unchanged
        assert spec.start_seconds == 100.0
        assert spec.end_seconds == 110.0
        # Analysis range expanded by context
        assert spec.analysis_start_seconds == 98.0
        assert spec.analysis_end_seconds == 112.0
        assert spec.analysis_start_frame < spec.start_frame
        assert spec.analysis_end_frame > spec.end_frame

    def test_context_clamped_at_zero(self) -> None:
        """Context should not go before time 0."""
        spec = build_range_spec(
            start=1.0, end=5.0, context=3.0,
            fps=24.0, total_frames=172800, total_duration=7200.0,
            frame_offset=0,
        )
        assert spec.analysis_start_seconds == 0.0
        assert spec.analysis_start_frame == 0

    def test_inconsistent_end_duration_raises(self) -> None:
        """--start + --end + --duration inconsistent → error."""
        with pytest.raises(RangeError, match="Inconsistent"):
            build_range_spec(
                start=100.0, end=130.0, duration=50.0,  # 100+50=150 ≠ 130
                fps=24.0, total_frames=172800, total_duration=7200.0,
                frame_offset=0,
            )

    def test_end_before_start_raises(self) -> None:
        """--end before --start → error."""
        with pytest.raises(RangeError, match="must be after start"):
            build_range_spec(
                start=200.0, end=100.0,
                fps=24.0, total_frames=172800, total_duration=7200.0,
                frame_offset=0,
            )

    def test_negative_start_raises(self) -> None:
        """Negative start → error."""
        with pytest.raises(RangeError, match="cannot be negative"):
            build_range_spec(
                start=-10.0, end=100.0,
                fps=24.0, total_frames=172800, total_duration=7200.0,
                frame_offset=0,
            )

    def test_end_exceeds_duration_raises(self) -> None:
        """End beyond source duration → error."""
        with pytest.raises(RangeError, match="exceeds source duration"):
            build_range_spec(
                start=100.0, end=8000.0,
                fps=24.0, total_frames=172800, total_duration=7200.0,
                frame_offset=0,
            )

    def test_serialization_round_trip(self) -> None:
        """to_dict/from_dict should preserve all fields."""
        spec = build_range_spec(
            start=100.0, end=130.0, context=2.0,
            fps=24.0, total_frames=172800, total_duration=7200.0,
            frame_offset=58,
        )
        data = spec.to_dict()
        restored = RangeSpec.from_dict(data)
        assert restored.start_seconds == spec.start_seconds
        assert restored.end_seconds == spec.end_seconds
        assert restored.start_frame == spec.start_frame
        assert restored.end_frame == spec.end_frame
        assert restored.om_start_frame == spec.om_start_frame
        assert restored.om_end_frame == spec.om_end_frame
        assert restored.context_seconds == spec.context_seconds
        assert restored.is_full_range == spec.is_full_range
