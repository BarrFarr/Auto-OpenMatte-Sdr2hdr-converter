"""Tests for shot detection (FEAT-004).

Covers:
- Hard cut detection
- Fade to/from black
- Dissolve
- No false positives on static/black scenes
- Adaptive threshold behavior
- min_shot_frames enforcement
- OM mapping via sync offset
- frame_range limiting
- Determinism
- Batch extraction (not per-frame)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.shots import (
    _classify_transitions,
    _compute_adaptive_threshold,
    _compute_histogram,
    _histogram_chi_square,
    _is_low_information_boundary,
    compute_boundary_score,
    detect_shots,
)
from auto_openmatte.core.config import ShotConfig
from auto_openmatte.core.models import (
    SourceInfo,
    SyncModel,
    SyncStatus,
    VideoStreamInfo,
)
from auto_openmatte.utils.frames import compute_edge_map


def _make_source(
    frame_count: int = 1000,
    fps: float = 24.0,
) -> SourceInfo:
    stream = VideoStreamInfo(
        index=0,
        codec="hevc",
        width=3840,
        height=1600,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )
    return SourceInfo(path=Path("test.mkv"), selected_stream=stream)


def _make_sync(offset: int = 1167) -> SyncModel:
    return SyncModel(
        frame_offset=offset,
        confidence=0.97,
        status=SyncStatus.LOCKED,
        frame_locked=True,
    )


# ---------------------------------------------------------------------------
# Test: histogram metrics
# ---------------------------------------------------------------------------


class TestHistogramMetrics:
    """Tests for histogram computation and comparison."""

    def test_identical_histograms(self) -> None:
        """Same frame → chi-square = 0."""
        frame = np.random.default_rng(42).random((100, 100))
        h = _compute_histogram(frame)
        assert _histogram_chi_square(h, h) == pytest.approx(0.0)

    def test_different_histograms(self) -> None:
        """Completely different frames → high chi-square."""
        dark = np.zeros((100, 100)) + 0.1
        bright = np.ones((100, 100)) * 0.9
        h_dark = _compute_histogram(dark)
        h_bright = _compute_histogram(bright)
        chi2 = _histogram_chi_square(h_dark, h_bright)
        assert chi2 > 1.0

    def test_histogram_normalized(self) -> None:
        """Histogram should sum to 1.0."""
        frame = np.random.default_rng(1).random((50, 50))
        h = _compute_histogram(frame)
        assert h.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: boundary score
# ---------------------------------------------------------------------------


class TestBoundaryScore:
    """Tests for combined boundary score."""

    def test_identical_frames_low_score(self) -> None:
        """Identical frames → near-zero boundary score."""
        frame = np.random.default_rng(42).random((80, 80))
        edge = compute_edge_map(frame)
        combined, _, _, _ = compute_boundary_score(frame, frame, edge, edge)
        assert combined < 0.01

    def test_different_frames_high_score(self) -> None:
        """Completely different frames → high boundary score."""
        rng = np.random.default_rng(42)
        frame_a = rng.random((80, 80))
        frame_b = rng.random((80, 80))
        edge_a = compute_edge_map(frame_a)
        edge_b = compute_edge_map(frame_b)
        combined, _, _, _ = compute_boundary_score(frame_a, frame_b, edge_a, edge_b)
        assert combined > 0.2

    def test_hard_cut_simulation(self) -> None:
        """Simulate hard cut: structured scene → completely different scene."""
        # Scene A: horizontal bars
        frame_a = np.zeros((80, 80))
        frame_a[20:40, :] = 0.8
        frame_a[60:80, :] = 0.5
        # Scene B: vertical bars
        frame_b = np.zeros((80, 80))
        frame_b[:, 10:30] = 0.7
        frame_b[:, 50:70] = 0.6

        edge_a = compute_edge_map(frame_a)
        edge_b = compute_edge_map(frame_b)
        combined, _, _, _ = compute_boundary_score(frame_a, frame_b, edge_a, edge_b)
        assert combined > 0.15


# ---------------------------------------------------------------------------
# Test: low-information suppression
# ---------------------------------------------------------------------------


class TestLowInformation:
    """Tests for low-information boundary suppression."""

    def test_black_to_black_suppressed(self) -> None:
        """Both frames near-black → suppressed."""
        black_a = np.zeros((80, 80)) + 0.005
        black_b = np.zeros((80, 80)) + 0.01
        edge_a = compute_edge_map(black_a)
        edge_b = compute_edge_map(black_b)
        # Both have mean luminance < 0.03 → suppressed
        assert _is_low_information_boundary(black_a, black_b, edge_a, edge_b)

    def test_scene_to_black_not_suppressed(self) -> None:
        """Scene → black is a valid transition, NOT suppressed."""
        scene = np.random.default_rng(42).random((80, 80)) * 0.5 + 0.3
        black = np.zeros((80, 80)) + 0.005
        edge_scene = compute_edge_map(scene)
        edge_black = compute_edge_map(black)
        assert not _is_low_information_boundary(scene, black, edge_scene, edge_black)

    def test_black_to_scene_not_suppressed(self) -> None:
        """Black → scene is a valid transition, NOT suppressed."""
        black = np.zeros((80, 80)) + 0.005
        scene = np.random.default_rng(42).random((80, 80)) * 0.5 + 0.3
        edge_black = compute_edge_map(black)
        edge_scene = compute_edge_map(scene)
        assert not _is_low_information_boundary(black, scene, edge_black, edge_scene)


# ---------------------------------------------------------------------------
# Test: adaptive threshold
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    """Tests for adaptive threshold computation."""

    def test_low_variance_material(self) -> None:
        """Mostly static material → low threshold."""
        scores = np.array([0.01, 0.02, 0.01, 0.015, 0.01, 0.02, 0.015, 0.01])
        threshold = _compute_adaptive_threshold(scores, multiplier=3.0)
        # mean ≈ 0.014, std ≈ 0.004 → threshold ≈ 0.026
        assert threshold < 0.05
        assert threshold > 0.02  # Minimum floor

    def test_spike_excluded_from_baseline(self) -> None:
        """A single spike (real cut) should not inflate the threshold."""
        # 99 low scores + 1 spike → spike is in top 5% and excluded
        scores = np.array([0.01, 0.02, 0.01, 0.015, 0.01, 0.02, 0.015, 0.01] * 12 + [0.8])
        threshold = _compute_adaptive_threshold(scores, multiplier=3.0)
        # The 0.8 spike should be in top 5% and excluded
        # Baseline mean ≈ 0.014, std ≈ 0.004 → threshold ≈ 0.026
        assert threshold < 0.10

    def test_empty_scores(self) -> None:
        """Empty scores → high threshold (no detections)."""
        threshold = _compute_adaptive_threshold(np.array([]), multiplier=3.0)
        assert threshold == 1.0


# ---------------------------------------------------------------------------
# Test: transition classification
# ---------------------------------------------------------------------------


class TestClassifyTransitions:
    """Tests for transition type classification."""

    def test_hard_cut_detected(self) -> None:
        """Single high spike → hard cut."""
        scores = np.array([0.01, 0.02, 0.01, 0.8, 0.01, 0.02, 0.01])
        mean_lums = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        threshold = 0.1
        boundaries = _classify_transitions(scores, threshold, mean_lums)
        assert len(boundaries) == 1
        assert boundaries[0][1] == "hard"
        assert boundaries[0][0] == 4  # frame index of new shot

    def test_fade_to_black(self) -> None:
        """Gradual increase with luminance dropping to near-zero → fade."""
        n = 20
        scores = np.zeros(n)
        scores[8:15] = np.linspace(0.05, 0.15, 7)  # Gradual elevation
        mean_lums = np.ones(n + 1) * 0.4
        mean_lums[12:] = 0.02  # Drops to black
        threshold = 0.08
        boundaries = _classify_transitions(scores, threshold, mean_lums)
        # Should detect a fade
        fades = [b for b in boundaries if b[1] == "fade"]
        assert len(fades) >= 1

    def test_dissolve(self) -> None:
        """Gradual elevation without black → dissolve."""
        n = 20
        scores = np.zeros(n)
        scores[5:12] = np.array([0.06, 0.08, 0.10, 0.12, 0.10, 0.08, 0.06])
        mean_lums = np.ones(n + 1) * 0.5  # No trend to black
        threshold = 0.08
        boundaries = _classify_transitions(
            scores, threshold, mean_lums, detect_transitions=True
        )
        dissolves = [b for b in boundaries if b[1] == "dissolve"]
        assert len(dissolves) >= 1

    def test_static_no_detection(self) -> None:
        """All low scores → no boundaries detected."""
        scores = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        mean_lums = np.ones(6) * 0.5
        threshold = 0.1
        boundaries = _classify_transitions(scores, threshold, mean_lums)
        assert len(boundaries) == 0

    def test_min_shot_frames_enforced(self) -> None:
        """Two cuts too close together → one suppressed."""
        scores = np.zeros(20)
        scores[3] = 0.9  # Cut at frame 3
        scores[5] = 0.8  # Cut at frame 5 — only 2 frames apart
        mean_lums = np.ones(21) * 0.5
        threshold = 0.1
        boundaries = _classify_transitions(
            scores, threshold, mean_lums, min_shot_frames=6
        )
        # Only one should survive
        assert len(boundaries) == 1
        # The higher-score cut (0.9 at frame 3) should win
        assert boundaries[0][0] == 4  # frame index = 3 + 1


# ---------------------------------------------------------------------------
# Test: detect_shots integration (mocked extraction)
# ---------------------------------------------------------------------------


class TestDetectShotsIntegration:
    """Integration tests for detect_shots with mocked frame extraction."""

    def _mock_frames_with_cut(self, n_frames: int, cut_at: int):
        """Create mock frame list with a hard cut at specified position."""
        rng = np.random.default_rng(42)
        frames = []
        for i in range(n_frames):
            if i < cut_at:
                # Scene A: dark structured scene
                f = np.zeros((200, 480))
                f[50:100, 100:200] = 0.6
                f += rng.random((200, 480)) * 0.02
            else:
                # Scene B: bright different scene
                f = np.ones((200, 480)) * 0.4
                f[80:150, 200:400] = 0.8
                f += rng.random((200, 480)) * 0.02
            frames.append(f)
        return frames

    def test_single_hard_cut(self) -> None:
        """Single hard cut in middle of clip → 2 shots detected."""
        source = _make_source(frame_count=100, fps=24.0)
        sync = _make_sync(offset=1167)
        cut_frame = 50
        mock_frames = self._mock_frames_with_cut(100, cut_frame)

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(source, sync, config=ShotConfig())

        # Should detect at least 2 shots
        assert len(shots) >= 2
        # Verify OM mapping uses offset
        for shot in shots:
            assert shot.om_start_frame == shot.hdr_start_frame + 1167
            assert shot.om_end_frame == shot.hdr_end_frame + 1167

    def test_static_scene_single_shot(self) -> None:
        """Static scene with no cuts → 1 shot."""
        source = _make_source(frame_count=100, fps=24.0)
        sync = _make_sync(offset=50)

        # All frames are similar
        rng = np.random.default_rng(42)
        base = np.zeros((200, 480))
        base[50:150, 100:400] = 0.5
        mock_frames = [base + rng.random((200, 480)) * 0.01 for _ in range(100)]

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(source, sync, config=ShotConfig())

        assert len(shots) == 1
        assert shots[0].hdr_start_frame == 0
        assert shots[0].hdr_end_frame == 100
        assert shots[0].om_start_frame == 50
        assert shots[0].om_end_frame == 150

    def test_frame_range_respected(self) -> None:
        """Shot detection limited to frame_range."""
        source = _make_source(frame_count=1000, fps=24.0)
        sync = _make_sync(offset=100)

        # Mock: static content
        mock_frames = [np.ones((200, 480)) * 0.5 for _ in range(200)]

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(
                source, sync,
                config=ShotConfig(),
                frame_range=(500, 700),
            )

        # All shots should be within the range
        for shot in shots:
            assert shot.hdr_start_frame >= 500
            assert shot.hdr_end_frame <= 700

    def test_om_mapping_uses_sync_offset(self) -> None:
        """Verify OM frames = HDR frames + sync_model.frame_offset."""
        source = _make_source(frame_count=100, fps=24.0)
        offset = 58
        sync = _make_sync(offset=offset)

        mock_frames = [np.ones((200, 480)) * 0.5 for _ in range(100)]

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(source, sync, config=ShotConfig())

        for shot in shots:
            assert shot.om_start_frame - shot.hdr_start_frame == offset
            assert shot.om_end_frame - shot.hdr_end_frame == offset

    def test_empty_source(self) -> None:
        """Source with 0 frames → empty list."""
        source = _make_source(frame_count=0)
        sync = _make_sync()
        shots = detect_shots(source, sync)
        assert shots == []

    def test_deterministic(self) -> None:
        """Same input → same output."""
        source = _make_source(frame_count=100, fps=24.0)
        sync = _make_sync(offset=1167)
        cut_frame = 40
        mock_frames = self._mock_frames_with_cut(100, cut_frame)

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots_1 = detect_shots(source, sync, config=ShotConfig())

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots_2 = detect_shots(source, sync, config=ShotConfig())

        assert len(shots_1) == len(shots_2)
        for s1, s2 in zip(shots_1, shots_2):
            assert s1.hdr_start_frame == s2.hdr_start_frame
            assert s1.hdr_end_frame == s2.hdr_end_frame
            assert s1.cut_type == s2.cut_type

    def test_batch_extraction_not_per_frame(self) -> None:
        """Verify extraction uses segments, not per-frame calls."""
        source = _make_source(frame_count=100, fps=24.0)
        sync = _make_sync()
        call_count = [0]

        def counting_extract(*args, **kwargs):
            call_count[0] += 1
            return [np.ones((200, 480)) * 0.5] * kwargs.get("n_frames", 100)

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            side_effect=counting_extract,
        ):
            detect_shots(source, sync, config=ShotConfig())

        # 100 frames < _SEGMENT_FRAMES (500) → should be 1 call
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Test: black scenes (Blade Runner 2049 characteristic)
# ---------------------------------------------------------------------------


class TestDarkScenes:
    """Tests for handling very dark content (BR2049 characteristic)."""

    def test_dark_scene_no_false_cuts(self) -> None:
        """Very dark scene with slight noise → no false cuts."""
        source = _make_source(frame_count=100, fps=24.0)
        sync = _make_sync()

        rng = np.random.default_rng(42)
        # Very dark frames with slight variation — mean lum ≈ 0.015
        mock_frames = [
            np.clip(rng.random((200, 480)) * 0.02 + 0.005, 0, 1) for _ in range(100)
        ]

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(source, sync, config=ShotConfig())

        # Should be 1 shot (no false cuts in dark content)
        assert len(shots) == 1

    def test_scene_to_black_detected(self) -> None:
        """Normal scene → fade to black should be detected as boundary."""
        source = _make_source(frame_count=50, fps=24.0)
        sync = _make_sync()

        rng = np.random.default_rng(42)
        mock_frames = []
        for i in range(50):
            if i < 25:
                # Normal scene
                f = rng.random((200, 480)) * 0.5 + 0.3
            else:
                # Black
                f = np.zeros((200, 480)) + 0.01
            mock_frames.append(f)

        with patch(
            "auto_openmatte.analysis.shots.extract_segment_at_time_grayscale",
            return_value=mock_frames,
        ):
            shots = detect_shots(source, sync, config=ShotConfig())

        # Should detect the transition
        assert len(shots) >= 2
