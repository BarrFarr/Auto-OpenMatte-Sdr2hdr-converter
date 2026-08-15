"""Tests for multi-window consensus and batch extraction architecture.

Tests:
1. All windows agree → LOCKED
2. Majority agree, one outlier → still LOCKED
3. Mixed invalids + valid consensus → LOCKED
4. All disagree → NOT LOCKED
5. All low-information → NOT LOCKED
6. Batch extraction uses single FFmpeg process (subprocess count check)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.sync import (
    _multi_window_consensus,
    find_global_offset,
)
from auto_openmatte.core.models import (
    FrameRateType,
    SourceInfo,
    SyncStatus,
    VideoStreamInfo,
)


def _make_source(
    path: str = "test.mkv",
    frame_count: int = 100000,
    fps: float = 24.0,
) -> SourceInfo:
    """Build a minimal SourceInfo for testing."""
    stream = VideoStreamInfo(
        index=0,
        codec="hevc",
        width=3840,
        height=2160,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps if fps > 0 else 0.0,
        frame_rate_type=FrameRateType.CFR,
    )
    return SourceInfo(path=Path(path), selected_stream=stream)


# ---------------------------------------------------------------------------
# Test 1: All windows agree
# ---------------------------------------------------------------------------


class TestMultiWindowConsensus:
    """Tests for _multi_window_consensus function."""

    def test_all_agree_locked(self) -> None:
        """5 windows all reporting +58 → offset=58, high confidence."""
        results = [
            (58, 0.97, 2.1),
            (58, 0.98, 2.3),
            (58, 0.96, 1.9),
            (58, 0.99, 2.5),
            (58, 0.97, 2.0),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert offset == 58
        assert confidence > 0.90

    def test_majority_agree_one_outlier(self) -> None:
        """4/5 agree on +58, one reports +59 → offset=58."""
        results = [
            (58, 0.97, 2.1),
            (58, 0.98, 2.3),
            (58, 0.96, 1.9),
            (59, 0.85, 1.5),  # Outlier
            (58, 0.97, 2.0),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert offset == 58
        assert confidence > 0.7

    def test_invalid_windows_with_consensus(self) -> None:
        """3 valid agree, 2 invalid (low score) → offset from valid."""
        results = [
            (58, 0.97, 2.1),
            (0, -1.0, 1.0),   # Invalid (low-information)
            (58, 0.96, 1.9),
            (0, -1.0, 1.0),   # Invalid
            (58, 0.98, 2.0),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert offset == 58
        # Confidence penalized due to few valid windows
        assert confidence > 0.5

    def test_all_disagree_low_confidence(self) -> None:
        """All windows report different offsets → low confidence."""
        results = [
            (12, 0.70, 1.2),
            (58, 0.65, 1.1),
            (-37, 0.60, 1.3),
            (91, 0.55, 1.1),
            (24, 0.72, 1.2),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # Each offset has count=1, so confidence should be very low
        # agreement_ratio = 1/5 = 0.2, mean_score ~ 0.65 → confidence ~ 0.13
        assert confidence < 0.5

    def test_all_low_information(self) -> None:
        """All windows return negative score (low-info) → confidence 0."""
        results = [
            (0, -1.0, 1.0),
            (0, -1.0, 1.0),
            (0, -1.0, 1.0),
            (0, -1.0, 1.0),
            (0, -1.0, 1.0),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Test: find_global_offset with multi-window (mocked extraction)
# ---------------------------------------------------------------------------


class TestFindGlobalOffsetMultiWindow:
    """Integration tests for multi-window find_global_offset."""

    def test_consistent_windows_locked(self) -> None:
        """All windows finding same offset → LOCKED."""
        hdr_source = _make_source(frame_count=100000, fps=24.0)
        om_source = _make_source(path="om.mkv", frame_count=100000, fps=24.0)

        # Mock _estimate_offset_single_window to return consistent results
        with patch(
            "auto_openmatte.analysis.sync._estimate_offset_single_window",
            return_value=(58, 0.97, 2.0),
        ):
            model = find_global_offset(hdr_source, om_source)

        assert model.frame_offset == 58
        assert model.status == SyncStatus.LOCKED
        assert model.frame_locked is True
        assert model.confidence > 0.9

    def test_all_flat_windows_failed(self) -> None:
        """All windows returning low-info → FAILED."""
        hdr_source = _make_source(frame_count=100000, fps=24.0)
        om_source = _make_source(path="om.mkv", frame_count=100000, fps=24.0)

        with patch(
            "auto_openmatte.analysis.sync._estimate_offset_single_window",
            return_value=(0, -1.0, 1.0),
        ):
            model = find_global_offset(hdr_source, om_source)

        assert model.status == SyncStatus.FAILED
        assert model.frame_locked is False

    def test_disagreeing_windows_failed(self) -> None:
        """Windows with scattered offsets → FAILED (low confidence)."""
        hdr_source = _make_source(frame_count=100000, fps=24.0)
        om_source = _make_source(path="om.mkv", frame_count=100000, fps=24.0)

        call_count = [0]
        offsets = [12, 58, -37, 91, 24]

        def varying_offset(*args, **kwargs):
            idx = call_count[0] % len(offsets)
            call_count[0] += 1
            return (offsets[idx], 0.6, 1.2)

        with patch(
            "auto_openmatte.analysis.sync._estimate_offset_single_window",
            side_effect=varying_offset,
        ):
            model = find_global_offset(hdr_source, om_source)

        assert model.status == SyncStatus.FAILED
        assert model.frame_locked is False


# ---------------------------------------------------------------------------
# Test: Batch extraction architecture (subprocess count)
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Verify that batch extraction uses single FFmpeg process per segment."""

    def test_segment_extraction_single_subprocess(self) -> None:
        """extract_segment_at_time_grayscale should call subprocess.run ONCE."""
        from auto_openmatte.utils.frames import extract_segment_at_time_grayscale

        call_count = [0]

        def counting_run(*args, **kwargs):
            call_count[0] += 1
            # Return empty result (no actual ffmpeg)
            from unittest.mock import MagicMock

            result = MagicMock()
            result.returncode = 1
            result.stdout = b""
            return result

        with patch("subprocess.run", side_effect=counting_run):
            extract_segment_at_time_grayscale(
                Path("fake.mkv"), start_seconds=10.0, n_frames=100, width=480
            )

        # Should be exactly ONE subprocess call for 100 frames
        assert call_count[0] == 1

    def test_build_diff_signal_batch_single_subprocess(self) -> None:
        """_build_diff_signal_batch should call subprocess only ONCE."""
        from auto_openmatte.analysis.sync import _build_diff_signal_batch

        source = _make_source(frame_count=10000)
        call_count = [0]

        def counting_run(*args, **kwargs):
            call_count[0] += 1
            from unittest.mock import MagicMock

            result = MagicMock()
            result.returncode = 1
            result.stdout = b""
            return result

        with patch("subprocess.run", side_effect=counting_run):
            _build_diff_signal_batch(source, start_seconds=5.0, n_frames=360)

        # Single FFmpeg call for the entire segment
        assert call_count[0] == 1

    def test_find_global_offset_limited_subprocesses(self) -> None:
        """find_global_offset with 5 windows should use ~10 subprocess calls total."""
        hdr_source = _make_source(frame_count=100000)
        om_source = _make_source(path="om.mkv", frame_count=100000)
        call_count = [0]

        def counting_run(*args, **kwargs):
            call_count[0] += 1
            from unittest.mock import MagicMock

            result = MagicMock()
            result.returncode = 1
            result.stdout = b""
            return result

        with patch("subprocess.run", side_effect=counting_run):
            model = find_global_offset(hdr_source, om_source)

        # 5 windows × 2 sources = 10 FFmpeg calls (not 9000+)
        assert call_count[0] == 10
        # With no actual frames, should fail gracefully
        assert model.status == SyncStatus.FAILED


# ---------------------------------------------------------------------------
# Tests: ±1 frame tolerance in consensus grouping
# ---------------------------------------------------------------------------


class TestConsensusGrouping:
    """Tests for ±1 frame tolerance in _multi_window_consensus."""

    def test_1166_1167_1167_1167_one_group(self) -> None:
        """[1166, 1167, 1167, 1167] should form ONE group → offset 1167."""
        results = [
            (1166, 0.99, 1.15),
            (1167, 0.98, 1.18),
            (1167, 0.97, 1.07),
            (1167, 0.96, 1.02),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # All within ±1 of each other → one group, majority 1167
        assert offset == 1167
        # 4/4 agree → confidence should be high
        assert confidence > 0.90

    def test_58_59_58_59_one_group(self) -> None:
        """[58, 59, 58, 59] should form ONE group."""
        results = [
            (58, 0.97, 2.0),
            (59, 0.96, 1.9),
            (58, 0.98, 2.1),
            (59, 0.95, 1.8),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # All within ±1 → one group
        # 58 has higher total score (0.97+0.98=1.95 vs 0.96+0.95=1.91)
        assert offset == 58
        assert confidence > 0.90

    def test_58_60_boundary(self) -> None:
        """[58, 60, 58, 58] — 60 is NOT within ±1 of 58."""
        results = [
            (58, 0.97, 2.0),
            (60, 0.85, 1.5),
            (58, 0.98, 2.1),
            (58, 0.96, 1.9),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # 58 group has 3 members, 60 is separate (diff=2, not ±1)
        assert offset == 58
        # 3/4 valid agree
        assert confidence > 0.7

    def test_no_chain_grouping_58_59_60(self) -> None:
        """[58, 59, 60] must NOT chain: 58~59 and 59~60 does NOT mean 58~60."""
        results = [
            (58, 0.90, 1.5),
            (59, 0.95, 1.8),  # Highest score → becomes representative
            (60, 0.85, 1.4),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # 59 has highest score → establishes group first
        # 58 is within ±1 of 59 → joins group
        # 60 is within ±1 of 59 → joins group
        # All in one group (representative=59)
        # Best offset within group: 59 (highest score)
        assert offset == 59

    def test_no_chain_grouping_100_101_102_103(self) -> None:
        """[100, 101, 102, 103] — prevents unbounded chain."""
        results = [
            (100, 0.90, 1.5),
            (101, 0.92, 1.6),
            (102, 0.91, 1.5),
            (103, 0.89, 1.4),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # 101 has highest score → rep for group 1
        # 100 within ±1 of 101 → group 1
        # 102 within ±1 of 101 → group 1
        # 103: NOT within ±1 of 101 (diff=2) → new group
        # Group 1: [100, 101, 102] — 3 members
        # Group 2: [103] — 1 member
        # Winner: group 1, best offset = 101
        assert offset == 101

    def test_all_valid_agree_high_confidence(self) -> None:
        """All windows agree exactly → highest confidence."""
        results = [
            (58, 0.99, 2.5),
            (58, 0.98, 2.3),
            (58, 0.97, 2.1),
            (58, 0.96, 2.0),
            (58, 0.95, 1.9),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert offset == 58
        assert confidence > 0.95

    def test_one_pm1_outlier_still_high_confidence(self) -> None:
        """One ±1 outlier should not destroy consensus."""
        results = [
            (1167, 0.98, 2.0),
            (1167, 0.97, 1.9),
            (1166, 0.99, 1.5),  # ±1 outlier — still in group
            (1167, 0.96, 1.8),
            (0, -1.0, 1.0),     # Invalid
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # 4 valid windows, all in same ±1 group
        assert offset == 1167
        assert confidence > 0.90

    def test_truly_inconsistent_still_fails(self) -> None:
        """Genuinely different offsets should still fail."""
        results = [
            (50, 0.80, 1.3),
            (200, 0.75, 1.2),
            (500, 0.70, 1.1),
            (1000, 0.65, 1.0),
            (3000, 0.60, 1.0),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        # Each is its own group, max group size = 1
        # agreement_ratio = 1/5 = 0.2
        assert confidence < 0.5

    def test_low_info_windows_not_in_consensus(self) -> None:
        """Windows with score <= 0.3 must not participate in consensus."""
        results = [
            (58, 0.98, 2.0),
            (58, 0.97, 1.9),
            (999, 0.10, 1.0),  # Low-info — excluded
            (999, -1.0, 1.0),  # Invalid — excluded
            (58, 0.96, 1.8),
        ]
        offset, confidence = _multi_window_consensus(results, min_confidence=0.95)
        assert offset == 58
        # 3/3 valid agree → high confidence
        assert confidence > 0.90
