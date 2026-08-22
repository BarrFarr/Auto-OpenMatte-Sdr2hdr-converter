"""Tests for synchronization algorithm internals.

Covers edge cases and key behaviors of the temporal sync engine:
- Known offset detection on synthetic signals
- Negative and zero offsets
- Confidence scoring
- Drift detection
- Low-information rejection
- HDR/SDR luminance domain differences not causing false mismatches
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.sync import (
    _MIN_DIFF_ENERGY,
    _center_crop,
    _compute_frame_energy,
    _compute_temporal_diff,
    _find_best_offset_ncc,
    _get_total_frames,
)
from auto_openmatte.core.config import SyncConfig
from auto_openmatte.core.exceptions import SyncDriftError
from auto_openmatte.core.models import (
    FrameRateType,
    SourceInfo,
    SyncModel,
    SyncStatus,
    VideoStreamInfo,
)

# ---------------------------------------------------------------------------
# Helper: build synthetic temporal-difference signals
# ---------------------------------------------------------------------------


def _make_signal_with_events(length: int, event_positions: list[int], seed: int = 42) -> np.ndarray:
    """Create a 1-D signal with peaks at specified positions.

    Background is low-amplitude noise; events are high peaks.
    This simulates scene-change temporal differences.
    """
    rng = np.random.default_rng(seed)
    signal = rng.random(length) * 0.01  # Low background
    for pos in event_positions:
        if 0 <= pos < length:
            signal[pos] = 0.5 + rng.random() * 0.3  # High peak
    return signal


def _make_source(
    path: str = "test.mkv",
    frame_count: int = 10000,
    fps: float = 24.0,
    width: int = 3840,
    height: int = 2160,
    frame_rate_type: FrameRateType = FrameRateType.CFR,
) -> SourceInfo:
    """Build a minimal SourceInfo for testing."""
    stream = VideoStreamInfo(
        index=0,
        codec="hevc",
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps if fps > 0 else 0.0,
        frame_rate_type=frame_rate_type,
    )
    source = SourceInfo(path=Path(path), selected_stream=stream)
    return source


# ---------------------------------------------------------------------------
# Tests: _find_best_offset_ncc with synthetic signals
# ---------------------------------------------------------------------------


class TestFindBestOffsetNCC:
    """Tests for the NCC-based offset finder on 1-D signals."""

    def test_known_positive_offset(self) -> None:
        """Detect a known positive offset between two signals."""
        events = [50, 120, 200, 350, 500, 680]
        offset = 37

        hdr_signal = _make_signal_with_events(800, events)
        # OM signal has same events shifted by +offset
        om_events = [e + offset for e in events]
        om_signal = _make_signal_with_events(850, om_events)

        detected_offset, score, _ = _find_best_offset_ncc(hdr_signal, om_signal, max_offset=60)
        assert detected_offset == offset
        assert score > 0.9

    def test_known_negative_offset(self) -> None:
        """Detect a known negative offset (OM starts before HDR)."""
        events = [100, 200, 300, 400, 500, 600]
        offset = -23

        hdr_signal = _make_signal_with_events(800, events)
        om_events = [e + offset for e in events]
        om_signal = _make_signal_with_events(800, om_events)

        detected_offset, score, _ = _find_best_offset_ncc(hdr_signal, om_signal, max_offset=50)
        assert detected_offset == offset
        assert score > 0.9

    def test_zero_offset(self) -> None:
        """Detect zero offset when signals are aligned."""
        events = [50, 150, 300, 450, 600]
        hdr_signal = _make_signal_with_events(700, events)
        om_signal = _make_signal_with_events(700, events, seed=43)  # Slight noise diff

        detected_offset, score, _ = _find_best_offset_ncc(hdr_signal, om_signal, max_offset=30)
        assert detected_offset == 0
        assert score > 0.8

    def test_high_confidence_for_strong_match(self) -> None:
        """Identical signals should produce confidence near 1.0."""
        events = [30, 80, 150, 250, 370, 500, 620]
        signal = _make_signal_with_events(700, events)

        detected_offset, score, _ = _find_best_offset_ncc(signal, signal, max_offset=20)
        assert detected_offset == 0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_low_confidence_for_unrelated_signals(self) -> None:
        """Unrelated signals should produce low confidence."""
        hdr_signal = _make_signal_with_events(500, [50, 150, 250, 350], seed=1)
        om_signal = _make_signal_with_events(500, [80, 200, 320, 450], seed=99)

        _, score, _ = _find_best_offset_ncc(hdr_signal, om_signal, max_offset=60)
        # Score should be notably lower than for matching signals
        assert score < 0.7

    def test_flat_signals_rejected(self) -> None:
        """Two flat (zero-variance) signals should not produce a high score."""
        flat = np.ones(200) * 0.001
        _, score, _ = _find_best_offset_ncc(flat, flat, max_offset=20)
        # Score should be -2.0 (no valid correlation found) since std < threshold
        assert score < 0.0


# ---------------------------------------------------------------------------
# Tests: _center_crop edge cases
# ---------------------------------------------------------------------------


class TestCenterCropEdgeCases:
    """Additional edge cases for center crop."""

    def test_crop_odd_dimensions(self) -> None:
        """Odd source with even target should crop correctly."""
        arr = np.arange(77).reshape(7, 11)
        cropped = _center_crop(arr, 4, 6)
        assert cropped.shape == (4, 6)

    def test_crop_1x1(self) -> None:
        """Cropping to 1x1 should return center pixel."""
        arr = np.arange(9).reshape(3, 3)
        cropped = _center_crop(arr, 1, 1)
        assert cropped.shape == (1, 1)
        assert cropped[0, 0] == arr[1, 1]

    def test_target_larger_than_source(self) -> None:
        """If target > source, should clamp to source size."""
        arr = np.arange(20).reshape(4, 5)
        cropped = _center_crop(arr, 10, 10)
        assert cropped.shape == (4, 5)
        np.testing.assert_array_equal(cropped, arr)


# ---------------------------------------------------------------------------
# Tests: _get_total_frames edge cases
# ---------------------------------------------------------------------------


class TestGetTotalFramesEdgeCases:
    """Edge cases for frame count determination."""

    def test_no_selected_stream(self) -> None:
        """Source without selected_stream should return 0."""
        source = SourceInfo(path=Path("test.mkv"), selected_stream=None)
        assert _get_total_frames(source) == 0

    def test_zero_frame_count_uses_duration(self) -> None:
        """frame_count=0 should fall back to duration * fps."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=1920, height=1080,
            frame_count=0, fps=30.0, duration_seconds=60.0,
        )
        source = SourceInfo(path=Path("test.mkv"), selected_stream=stream)
        # frame_count=0 is falsy, so fallback to 30*60=1800
        assert _get_total_frames(source) == 1800

    def test_negative_frame_count_uses_duration(self) -> None:
        """Negative frame_count should fall back to duration * fps."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=1920, height=1080,
            frame_count=-1, fps=24.0, duration_seconds=100.0,
        )
        source = SourceInfo(path=Path("test.mkv"), selected_stream=stream)
        # -1 is not > 0, so fallback
        assert _get_total_frames(source) == 2400


# ---------------------------------------------------------------------------
# Tests: validate_sync — drift detection
# ---------------------------------------------------------------------------


class TestValidateSyncDrift:
    """Tests for drift detection in validate_sync."""

    def test_stable_offset_passes(self) -> None:
        """Stable offset at all checkpoints should result in LOCKED."""
        from auto_openmatte.analysis.sync import validate_sync

        hdr_source = _make_source(frame_count=5000, fps=24.0)
        om_source = _make_source(path="om.mkv", frame_count=5000, fps=24.0)
        global_offset = 48

        sync_model = SyncModel(
            frame_offset=global_offset,
            confidence=0.98,
            status=SyncStatus.LOCKED,
            frame_locked=True,
        )

        config = SyncConfig(validation_points=10, max_drift_frames=0.5)

        # Mock _measure_local_offset_at_point to return consistent offset
        with patch(
            "auto_openmatte.analysis.sync._measure_local_offset_at_point",
            return_value=(global_offset, 0.95),
        ):
            result = validate_sync(hdr_source, om_source, sync_model, config=config)

        assert result.status == SyncStatus.LOCKED
        assert result.frame_locked is True
        assert result.drift_frames <= config.max_drift_frames

    def test_drifting_offset_raises(self) -> None:
        """Offset that varies > 0.5 frame should raise SyncDriftError."""
        from auto_openmatte.analysis.sync import validate_sync

        hdr_source = _make_source(frame_count=5000, fps=24.0)
        om_source = _make_source(path="om.mkv", frame_count=5000, fps=24.0)
        global_offset = 48

        sync_model = SyncModel(
            frame_offset=global_offset,
            confidence=0.98,
            status=SyncStatus.LOCKED,
            frame_locked=True,
        )

        config = SyncConfig(validation_points=10, max_drift_frames=0.5)

        # Mock: return an offset that drifts by 1 frame at some checkpoint
        call_count = [0]

        def drifting_offset(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 5:
                return (global_offset + 1, 0.9)  # Drift of 1 frame
            return (global_offset, 0.95)

        with patch(
            "auto_openmatte.analysis.sync._measure_local_offset_at_point",
            side_effect=drifting_offset,
        ):
            with pytest.raises(SyncDriftError):
                validate_sync(hdr_source, om_source, sync_model, config=config)


# ---------------------------------------------------------------------------
# Tests: compute helpers
# ---------------------------------------------------------------------------


class TestComputeHelpers:
    """Tests for internal compute functions."""

    def test_frame_energy_black(self) -> None:
        """Black frame should have zero edge energy."""
        black = np.zeros((100, 100), dtype=np.float64)
        assert _compute_frame_energy(black) == 0.0

    def test_frame_energy_edges(self) -> None:
        """Frame with strong edges should have positive energy."""
        frame = np.zeros((100, 100), dtype=np.float64)
        frame[40:60, :] = 1.0  # Horizontal bar
        from auto_openmatte.utils.frames import compute_edge_map

        edge = compute_edge_map(frame)
        energy = _compute_frame_energy(edge)
        assert energy > 0.01

    def test_temporal_diff_identical(self) -> None:
        """Identical edge maps should have zero temporal diff."""
        edge = np.random.default_rng(42).random((50, 50))
        diff = _compute_temporal_diff(edge, edge)
        assert diff == pytest.approx(0.0, abs=1e-10)

    def test_temporal_diff_different(self) -> None:
        """Different edge maps should have positive temporal diff."""
        rng = np.random.default_rng(42)
        a = rng.random((50, 50))
        b = rng.random((50, 50))
        diff = _compute_temporal_diff(a, b)
        assert diff > 0.1

    def test_temporal_diff_mismatched_sizes(self) -> None:
        """Different sized arrays should be center-cropped before comparison."""
        a = np.ones((60, 80), dtype=np.float64) * 0.5
        b = np.ones((50, 70), dtype=np.float64) * 0.5
        # Should not raise, and should return 0 since values are identical
        diff = _compute_temporal_diff(a, b)
        assert diff == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: HDR/SDR domain invariance
# ---------------------------------------------------------------------------


class TestDomainInvariance:
    """Edge-based fingerprints should not be affected by transfer function."""

    def test_pq_vs_gamma_same_structure(self) -> None:
        """Same spatial structure under PQ vs gamma should produce similar edges."""
        from auto_openmatte.utils.frames import compute_edge_map

        rng = np.random.default_rng(42)
        # Create a structured scene (gradients + edges)
        base = np.zeros((100, 100), dtype=np.float64)
        base[20:80, 20:80] = 0.5
        base[40:60, 40:60] = 0.8
        noise = rng.random((100, 100)) * 0.02
        scene = base + noise

        # Simulate "PQ encoded": apply PQ-like nonlinearity
        pq_scene = np.power(np.clip(scene, 0, 1), 0.25)  # Simplified PQ-like curve

        # Simulate "SDR gamma": apply gamma 2.2
        gamma_scene = np.power(np.clip(scene, 0, 1), 1.0 / 2.2)

        # Edge maps should be structurally similar despite different tonal curves
        pq_edges = compute_edge_map(pq_scene)
        gamma_edges = compute_edge_map(gamma_scene)

        # NCC between edge maps should be high
        from auto_openmatte.utils.math_utils import normalized_cross_correlation

        ncc = normalized_cross_correlation(pq_edges, gamma_edges)
        assert ncc > 0.85, (
            f"Edge maps under PQ vs gamma should correlate highly, got NCC={ncc:.3f}"
        )

    def test_temporal_diff_pq_vs_gamma(self) -> None:
        """Frame-to-frame temporal difference should be similar regardless of TF."""
        from auto_openmatte.utils.frames import compute_edge_map

        rng = np.random.default_rng(42)

        # Frame A: scene with objects
        frame_a = np.zeros((80, 80), dtype=np.float64)
        frame_a[10:30, 10:30] = 0.6
        frame_a[50:70, 50:70] = 0.4
        frame_a += rng.random((80, 80)) * 0.02

        # Frame B: objects moved
        frame_b = np.zeros((80, 80), dtype=np.float64)
        frame_b[15:35, 15:35] = 0.6
        frame_b[45:65, 45:65] = 0.4
        frame_b += rng.random((80, 80)) * 0.02

        # PQ versions
        pq_a = np.power(np.clip(frame_a, 0, 1), 0.25)
        pq_b = np.power(np.clip(frame_b, 0, 1), 0.25)

        # Gamma versions
        gamma_a = np.power(np.clip(frame_a, 0, 1), 1.0 / 2.2)
        gamma_b = np.power(np.clip(frame_b, 0, 1), 1.0 / 2.2)

        # Temporal diffs via edge maps
        diff_pq = _compute_temporal_diff(compute_edge_map(pq_a), compute_edge_map(pq_b))
        diff_gamma = _compute_temporal_diff(
            compute_edge_map(gamma_a), compute_edge_map(gamma_b)
        )

        # Both should detect meaningful change
        assert diff_pq > _MIN_DIFF_ENERGY
        assert diff_gamma > _MIN_DIFF_ENERGY

        # Relative magnitude should be in the same ballpark
        ratio = diff_pq / diff_gamma if diff_gamma > 0 else 0
        assert 0.3 < ratio < 3.0, (
            f"PQ/gamma temporal diffs should be comparable: {diff_pq:.4f} vs {diff_gamma:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: VFR handling
# ---------------------------------------------------------------------------


class TestVFRHandling:
    """VFR sources should be detected and handled safely."""

    def test_vfr_source_does_not_crash(self) -> None:
        """VFR source should not crash find_global_offset — it warns."""
        from auto_openmatte.analysis.sync import find_global_offset

        hdr_source = _make_source(frame_count=1000, frame_rate_type=FrameRateType.VFR)
        om_source = _make_source(path="om.mkv", frame_count=1000)

        # Mock batch extraction to return empty (simulating no ffmpeg)
        with patch(
            "auto_openmatte.analysis.sync._build_diff_signal_batch",
            return_value=np.zeros(100),
        ):
            # Should not crash — will produce low confidence due to flat signal
            model = find_global_offset(hdr_source, om_source)
            # Status should be FAILED due to low confidence on flat signal
            assert model.status == SyncStatus.FAILED
            assert model.frame_locked is False
