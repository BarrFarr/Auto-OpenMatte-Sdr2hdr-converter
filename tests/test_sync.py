"""Tests for synchronization logic."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.sync import _center_crop, _get_total_frames
from auto_openmatte.core.models import (
    SourceInfo,
    SyncModel,
    SyncStatus,
    VideoStreamInfo,
)
from auto_openmatte.utils.math_utils import normalized_cross_correlation


class TestCenterCrop:
    """Tests for center cropping utility."""

    def test_crop_smaller(self) -> None:
        """Crop to smaller size should center."""
        arr = np.arange(100).reshape(10, 10)
        cropped = _center_crop(arr, 6, 6)
        assert cropped.shape == (6, 6)
        # Should start from (2, 2)
        assert cropped[0, 0] == arr[2, 2]

    def test_crop_same_size(self) -> None:
        """Crop to same size should return unchanged."""
        arr = np.arange(100).reshape(10, 10)
        cropped = _center_crop(arr, 10, 10)
        assert cropped.shape == (10, 10)
        np.testing.assert_array_equal(cropped, arr)


class TestGetTotalFrames:
    """Tests for frame count determination."""

    def test_from_frame_count(self) -> None:
        """Should use frame_count if available."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=1920, height=1080,
            frame_count=1000, fps=24.0, duration_seconds=41.67,
        )
        source = SourceInfo(path=Path("test.mkv"), selected_stream=stream)
        assert _get_total_frames(source) == 1000

    def test_from_duration_fps(self) -> None:
        """Should compute from duration*fps if frame_count unavailable."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=1920, height=1080,
            frame_count=None, fps=24.0, duration_seconds=100.0,
        )
        source = SourceInfo(path=Path("test.mkv"), selected_stream=stream)
        assert _get_total_frames(source) == 2400


class TestNCC:
    """Tests for normalized cross correlation."""

    def test_identical_signals(self) -> None:
        """Identical arrays should have NCC = 1.0."""
        a = np.random.default_rng(42).random((100, 100))
        assert normalized_cross_correlation(a, a) == pytest.approx(1.0, abs=1e-10)

    def test_inverted_signals(self) -> None:
        """Inverted array should have NCC = -1.0."""
        a = np.random.default_rng(42).random((100, 100))
        b = 1.0 - a
        assert normalized_cross_correlation(a, b) == pytest.approx(-1.0, abs=1e-10)

    def test_uncorrelated(self) -> None:
        """Random unrelated arrays should have NCC near 0."""
        rng = np.random.default_rng(42)
        a = rng.random((100, 100))
        b = rng.random((100, 100))
        ncc = normalized_cross_correlation(a, b)
        assert abs(ncc) < 0.2  # Should be near zero

    def test_shifted_signal(self) -> None:
        """Shifted signal with same structure should have high NCC."""
        a = np.sin(np.linspace(0, 10, 10000)).reshape(100, 100)
        b = a + 5.0  # Constant offset doesn't change NCC
        assert normalized_cross_correlation(a, b) == pytest.approx(1.0, abs=1e-10)


class TestSyncModel:
    """Tests for SyncModel data structure."""

    def test_default_state(self) -> None:
        """Default SyncModel should be NOT_RUN."""
        model = SyncModel()
        assert model.status == SyncStatus.NOT_RUN
        assert model.frame_offset == 0
        assert model.frame_locked is False

    def test_locked_state(self) -> None:
        """Locked model should have all required fields."""
        model = SyncModel(
            frame_offset=58,
            confidence=0.998,
            status=SyncStatus.LOCKED,
            frame_locked=True,
            drift_frames=0.0,
            offset_seconds=58 / 23.976,
        )
        assert model.frame_locked is True
        assert model.frame_offset == 58
        assert model.status == SyncStatus.LOCKED
